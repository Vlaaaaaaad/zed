use std::ops::Range;
use std::rc::Rc;

use acp_thread::{AcpThread, AgentThreadEntry, AssistantMessageChunk, ThreadStatus};
use agent::ThreadStore;
use agent_client_protocol::schema::v1 as acp;
use agent_settings::AgentSettings;
use collections::{HashMap, HashSet};
use editor::{Editor, EditorEvent, EditorMode, MinimapVisibility, SizingBehavior};
use gpui::{
    AnyEntity, App, AppContext as _, Entity, EntityId, EventEmitter, FocusHandle, Focusable,
    ScrollHandle, SharedString, TextStyleRefinement, WeakEntity, Window,
};
use language::language_settings::SoftWrap;
use project::{AgentId, Project, project_settings::DiagnosticSeverity};
use rope::Point;
use settings::{Settings as _, ThinkingBlockDisplay, ToolCallDisplay};
use terminal_view::TerminalView;
use theme_settings::ThemeSettings;
use ui::{Context, IconName, TextSize};
use workspace::Workspace;

use crate::message_editor::{MessageEditor, MessageEditorEvent, SharedSessionCapabilities};

/// Maps an entry index through the removal of `removed` (a contiguous range of
/// entries), returning `None` if the index referred to a removed entry.
fn reindex_after_removal(index: usize, removed: &Range<usize>) -> Option<usize> {
    if index < removed.start {
        Some(index)
    } else if index < removed.end {
        None
    } else {
        Some(index - removed.len())
    }
}

/// Hint about what changed since the last bundle computation,
/// allowing a scoped recomputation of only the affected suffix.
#[derive(Clone, Debug)]
pub(crate) enum BundleRecomputeHint {
    /// A new entry was appended at the given index (always len-1).
    NewEntry(usize),
    /// An existing entry was updated at the given index.
    EntryUpdated(usize),
    /// A range of entries was removed.
    EntriesRemoved { range: Range<usize> },
}

/// Stable identifier for a bundle, surviving entry removals and reordering
/// when possible. Tool-call bundles are keyed by the first tool call's ID;
/// pure-thought bundles (which have no tool call ID) fall back to positional.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BundleId {
    ToolCall(acp::ToolCallId),
    Thoughts(usize),
}

const TOOL_KIND_COUNT: usize = 10;

#[derive(Clone, Debug)]
pub(crate) struct ToolCallBundle {
    pub start_index: usize,
    pub end_index: usize,
    /// Total tool call count. Used in tests; in production the pre-computed
    /// `label` field carries this information to the renderer.
    #[allow(dead_code)]
    pub tool_call_count: usize,
    #[allow(dead_code)]
    pub thought_count: usize,
    /// Per-kind tool call counts, indexed by `tool_kind_sort_key`.
    kind_counts: [u32; TOOL_KIND_COUNT],
    /// Thought chunks from the mixed message immediately after the bundle,
    /// pre-computed so rendering doesn't need to reach into the entries slice.
    #[allow(dead_code)]
    pub trailing_thought_count: usize,
    pub id: BundleId,
    pub label: SharedString,
    pub group_id: SharedString,
}

impl ToolCallBundle {
    /// Iterates over tool kinds that have a non-zero count, in sort order.
    pub fn kind_counts_iter(&self) -> impl Iterator<Item = (acp::ToolKind, u32)> + '_ {
        self.kind_counts
            .iter()
            .enumerate()
            .filter_map(|(key, &count)| {
                if count > 0 {
                    Some((tool_kind_from_sort_key(key), count))
                } else {
                    None
                }
            })
    }
}

fn is_entry_bundle_eligible(entry: &AgentThreadEntry) -> bool {
    match entry {
        AgentThreadEntry::ToolCall(tool_call) => {
            !tool_call.is_subagent()
                && !matches!(
                    tool_call.status,
                    acp_thread::ToolCallStatus::WaitingForConfirmation { .. }
                )
                && !matches!(tool_call.status, acp_thread::ToolCallStatus::Canceled)
        }
        AgentThreadEntry::AssistantMessage(msg) => {
            !msg.chunks.is_empty()
                && msg
                    .chunks
                    .iter()
                    .any(|chunk| matches!(chunk, AssistantMessageChunk::Thought { .. }))
        }
        _ => false,
    }
}

/// A terminating entry is counted into the current bundle but ends the run
/// afterwards. Mixed messages (thoughts + text) are terminating so output
/// text doesn't get swallowed into a mega-bundle.
fn is_entry_bundle_terminating(entry: &AgentThreadEntry) -> bool {
    matches!(entry, AgentThreadEntry::AssistantMessage(msg) if msg.chunks.iter().any(|c| {
        !matches!(c, AssistantMessageChunk::Thought { .. })
    }))
}

fn count_entry(
    entry: &AgentThreadEntry,
    tool_call_count: &mut usize,
    thought_count: &mut usize,
    kind_counts: &mut [u32; TOOL_KIND_COUNT],
    first_tool_call_id: &mut Option<acp::ToolCallId>,
) {
    match entry {
        AgentThreadEntry::ToolCall(tool_call) => {
            *tool_call_count += 1;
            kind_counts[tool_kind_sort_key(&tool_call.kind) as usize] += 1;
            if first_tool_call_id.is_none() {
                *first_tool_call_id = Some(tool_call.id.clone());
            }
        }
        AgentThreadEntry::AssistantMessage(msg) => {
            *thought_count += msg
                .chunks
                .iter()
                .filter(|c| matches!(c, AssistantMessageChunk::Thought { .. }))
                .count();
        }
        _ => {}
    }
}

fn trailing_thought_count(entries: &[AgentThreadEntry], at: usize) -> usize {
    if let Some(AgentThreadEntry::AssistantMessage(msg)) = entries.get(at) {
        msg.chunks
            .iter()
            .filter(|c| matches!(c, AssistantMessageChunk::Thought { .. }))
            .count()
    } else {
        0
    }
}

fn bundle_label(thought_count: usize, tool_call_count: usize) -> SharedString {
    match (thought_count, tool_call_count) {
        (0, 1) => SharedString::from("1 tool call"),
        (0, t) => SharedString::from(format!("{} tool calls", t)),
        (1, 0) => SharedString::from("1 thinking block"),
        (b, 0) => SharedString::from(format!("{} thinking blocks", b)),
        (1, 1) => SharedString::from("1 thinking block · 1 tool call"),
        (1, t) => SharedString::from(format!("1 thinking block · {} tool calls", t)),
        (b, 1) => SharedString::from(format!("{} thinking blocks · 1 tool call", b)),
        (b, t) => SharedString::from(format!("{} thinking blocks · {} tool calls", b, t)),
    }
}

pub(crate) fn tool_kind_info(kind: &acp::ToolKind) -> (u8, IconName) {
    match kind {
        acp::ToolKind::Read => (0, IconName::ToolSearch),
        acp::ToolKind::Edit => (1, IconName::ToolPencil),
        acp::ToolKind::Delete => (2, IconName::ToolDeleteFile),
        acp::ToolKind::Move => (3, IconName::ArrowRightLeft),
        acp::ToolKind::Search => (4, IconName::ToolSearch),
        acp::ToolKind::Execute => (5, IconName::ToolTerminal),
        acp::ToolKind::Think => (6, IconName::ToolThink),
        acp::ToolKind::Fetch => (7, IconName::ToolWeb),
        acp::ToolKind::SwitchMode => (8, IconName::ArrowRightLeft),
        acp::ToolKind::Other | _ => (9, IconName::ToolHammer),
    }
}

fn tool_kind_sort_key(kind: &acp::ToolKind) -> u8 {
    tool_kind_info(kind).0
}

fn tool_kind_from_sort_key(key: usize) -> acp::ToolKind {
    match key {
        0 => acp::ToolKind::Read,
        1 => acp::ToolKind::Edit,
        2 => acp::ToolKind::Delete,
        3 => acp::ToolKind::Move,
        4 => acp::ToolKind::Search,
        5 => acp::ToolKind::Execute,
        6 => acp::ToolKind::Think,
        7 => acp::ToolKind::Fetch,
        8 => acp::ToolKind::SwitchMode,
        _ => acp::ToolKind::Other,
    }
}

/// Compute bundles for a slice of entries, adding `offset` to all indices.
fn compute_tool_call_bundles(entries: &[AgentThreadEntry], offset: usize) -> Vec<ToolCallBundle> {
    let mut bundles = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        while i < entries.len() && !is_entry_bundle_eligible(&entries[i]) {
            i += 1;
        }
        if i >= entries.len() {
            break;
        }

        let run_start = i;
        let mut tool_call_count = 0;
        let mut thought_count = 0;
        let mut kind_counts = [0u32; TOOL_KIND_COUNT];
        let mut first_tool_call_id: Option<acp::ToolCallId> = None;

        while i < entries.len() && is_entry_bundle_eligible(&entries[i]) {
            count_entry(
                &entries[i],
                &mut tool_call_count,
                &mut thought_count,
                &mut kind_counts,
                &mut first_tool_call_id,
            );
            i += 1;
            if is_entry_bundle_terminating(&entries[i - 1]) {
                break;
            }
        }

        let run_len = i - run_start;
        if run_len > 1 {
            let start_index = run_start + offset;
            let end_index = i + offset;
            let id = match first_tool_call_id {
                Some(tc_id) => BundleId::ToolCall(tc_id),
                None => BundleId::Thoughts(start_index),
            };
            let total_thoughts = thought_count + trailing_thought_count(entries, i);
            bundles.push(ToolCallBundle {
                start_index,
                end_index,
                tool_call_count,
                thought_count,
                kind_counts,
                trailing_thought_count: trailing_thought_count(entries, i),
                label: bundle_label(total_thoughts, tool_call_count),
                id,
                group_id: SharedString::from(format!("bundle-header-{}", start_index)),
            });
        }
    }
    bundles
}

pub(crate) fn get_bundle_for_entry(
    bundles: &[ToolCallBundle],
    entry_ix: usize,
) -> Option<&ToolCallBundle> {
    let idx = bundles
        .binary_search_by_key(&entry_ix, |b| b.start_index)
        .unwrap_or_else(|err| err.saturating_sub(1));
    bundles
        .get(idx)
        .filter(|b| (b.start_index..b.end_index).contains(&entry_ix))
}

pub struct EntryViewState {
    workspace: WeakEntity<Workspace>,
    project: WeakEntity<Project>,
    thread_store: Option<Entity<ThreadStore>>,
    entries: Vec<Entry>,
    session_capabilities: SharedSessionCapabilities,
    agent_id: AgentId,
    expanded_thinking_blocks: HashSet<(usize, usize)>,
    auto_expanded_thinking_block: Option<(usize, usize)>,
    user_toggled_thinking_blocks: HashSet<(usize, usize)>,
    expanded_compactions: HashSet<usize>,
    expanded_tool_calls: HashSet<acp::ToolCallId>,
    tool_call_bundle_states: HashSet<BundleId>,
    /// Bundles the user explicitly collapsed during streaming, to prevent
    /// auto-expand from immediately re-opening them.
    user_collapsed_bundles: HashSet<BundleId>,
    /// The bundle currently auto-expanded during streaming, tracked so it
    /// can be compacted when streaming stops. `None` when no bundle is
    /// auto-expanded (or the user has taken over).
    auto_expanded_bundle: Option<BundleId>,
    bundles: Rc<[ToolCallBundle]>,
}

impl EntryViewState {
    pub fn new(
        workspace: WeakEntity<Workspace>,
        project: WeakEntity<Project>,
        thread_store: Option<Entity<ThreadStore>>,
        session_capabilities: SharedSessionCapabilities,
        agent_id: AgentId,
    ) -> Self {
        Self {
            workspace,
            project,
            thread_store,
            entries: Vec::new(),
            session_capabilities,
            agent_id,
            expanded_thinking_blocks: HashSet::default(),
            auto_expanded_thinking_block: None,
            user_toggled_thinking_blocks: HashSet::default(),
            expanded_compactions: HashSet::default(),
            expanded_tool_calls: HashSet::default(),
            tool_call_bundle_states: HashSet::default(),
            user_collapsed_bundles: HashSet::default(),
            auto_expanded_bundle: None,
            bundles: Rc::new([]),
        }
    }

    pub(crate) fn bundles(&self) -> &Rc<[ToolCallBundle]> {
        &self.bundles
    }

    pub(crate) fn is_tool_call_expanded(&self, tool_call_id: &acp::ToolCallId) -> bool {
        self.expanded_tool_calls.contains(tool_call_id)
    }

    pub(crate) fn expand_tool_call(&mut self, tool_call_id: acp::ToolCallId) {
        self.expanded_tool_calls.insert(tool_call_id);
    }

    pub(crate) fn collapse_tool_call(&mut self, tool_call_id: &acp::ToolCallId) {
        self.expanded_tool_calls.remove(tool_call_id);
    }

    pub(crate) fn toggle_tool_call_expansion(&mut self, tool_call_id: &acp::ToolCallId) {
        if !self.expanded_tool_calls.remove(tool_call_id) {
            self.expanded_tool_calls.insert(tool_call_id.clone());
        }
    }

    pub(crate) fn is_compaction_expanded(&self, entry_ix: usize) -> bool {
        self.expanded_compactions.contains(&entry_ix)
    }

    pub(crate) fn collapse_compaction(&mut self, entry_ix: usize) {
        self.expanded_compactions.remove(&entry_ix);
    }

    pub(crate) fn toggle_compaction_expansion(&mut self, entry_ix: usize) {
        if !self.expanded_compactions.remove(&entry_ix) {
            self.expanded_compactions.insert(entry_ix);
        }
    }

    pub(crate) fn clear_auto_expanded_thinking(&mut self) {
        self.auto_expanded_thinking_block = None;
    }

    pub(crate) fn is_auto_expanded_thinking_block(&self, key: (usize, usize)) -> bool {
        self.auto_expanded_thinking_block == Some(key)
    }

    pub(crate) fn auto_expand_streaming_thought(&mut self, thread: &AcpThread, cx: &App) -> bool {
        let thinking_display = AgentSettings::get_global(cx).thinking_display;

        if !matches!(
            thinking_display,
            ThinkingBlockDisplay::Auto | ThinkingBlockDisplay::Preview
        ) {
            return false;
        }

        let last_ix = thread.entries().len().saturating_sub(1);
        let key = match thread.entries().get(last_ix) {
            Some(AgentThreadEntry::AssistantMessage(message)) => match message.chunks.last() {
                Some(AssistantMessageChunk::Thought { .. }) => {
                    Some((last_ix, message.chunks.len() - 1))
                }
                _ => None,
            },
            _ => None,
        };

        if let Some(key) = key {
            if self.auto_expanded_thinking_block != Some(key) {
                self.auto_expanded_thinking_block = Some(key);
                self.expanded_thinking_blocks.insert(key);
                return true;
            }
        } else if self.auto_expanded_thinking_block.is_some() {
            if thinking_display == ThinkingBlockDisplay::Auto
                && let Some(key) = self.auto_expanded_thinking_block
                && !self.user_toggled_thinking_blocks.contains(&key)
            {
                self.expanded_thinking_blocks.remove(&key);
            }
            self.auto_expanded_thinking_block = None;
            return true;
        }

        false
    }

    pub(crate) fn toggle_thinking_block_expansion(&mut self, key: (usize, usize), cx: &App) {
        match AgentSettings::get_global(cx).thinking_display {
            ThinkingBlockDisplay::Auto => {
                let is_open = self.expanded_thinking_blocks.contains(&key)
                    || self.user_toggled_thinking_blocks.contains(&key);

                if is_open {
                    self.expanded_thinking_blocks.remove(&key);
                    self.user_toggled_thinking_blocks.remove(&key);
                } else {
                    self.expanded_thinking_blocks.insert(key);
                    self.user_toggled_thinking_blocks.insert(key);
                }
            }
            ThinkingBlockDisplay::Preview => {
                let is_user_expanded = self.user_toggled_thinking_blocks.contains(&key);
                let is_in_expanded_set = self.expanded_thinking_blocks.contains(&key);

                if is_user_expanded {
                    self.user_toggled_thinking_blocks.remove(&key);
                    self.expanded_thinking_blocks.remove(&key);
                } else if is_in_expanded_set {
                    self.user_toggled_thinking_blocks.insert(key);
                } else {
                    self.expanded_thinking_blocks.insert(key);
                    self.user_toggled_thinking_blocks.insert(key);
                }
            }
            ThinkingBlockDisplay::AlwaysExpanded => {
                if self.user_toggled_thinking_blocks.contains(&key) {
                    self.user_toggled_thinking_blocks.remove(&key);
                } else {
                    self.user_toggled_thinking_blocks.insert(key);
                }
            }
            ThinkingBlockDisplay::AlwaysCollapsed => {
                if self.user_toggled_thinking_blocks.contains(&key) {
                    self.user_toggled_thinking_blocks.remove(&key);
                    self.expanded_thinking_blocks.remove(&key);
                } else {
                    self.expanded_thinking_blocks.insert(key);
                    self.user_toggled_thinking_blocks.insert(key);
                }
            }
        }
    }

    pub(crate) fn thinking_block_state(&self, key: (usize, usize), cx: &App) -> (bool, bool) {
        let is_user_toggled = self.user_toggled_thinking_blocks.contains(&key);
        let is_in_expanded_set = self.expanded_thinking_blocks.contains(&key);

        match AgentSettings::get_global(cx).thinking_display {
            ThinkingBlockDisplay::Auto => {
                let is_open = is_user_toggled || is_in_expanded_set;
                (is_open, false)
            }
            ThinkingBlockDisplay::Preview => {
                let is_open = is_user_toggled || is_in_expanded_set;
                let is_constrained = is_in_expanded_set && !is_user_toggled;
                (is_open, is_constrained)
            }
            ThinkingBlockDisplay::AlwaysExpanded => (!is_user_toggled, false),
            ThinkingBlockDisplay::AlwaysCollapsed => (is_user_toggled, false),
        }
    }

    pub(crate) fn toggle_tool_call_bundle_expansion(&mut self, bundle_id: &BundleId, cx: &App) {
        match AgentSettings::get_global(cx).tool_call_display {
            ToolCallDisplay::Auto | ToolCallDisplay::Compact => {
                if self.tool_call_bundle_states.contains(bundle_id) {
                    self.tool_call_bundle_states.remove(bundle_id);
                    if self.auto_expanded_bundle.as_ref() == Some(bundle_id) {
                        self.auto_expanded_bundle = None;
                        self.user_collapsed_bundles.insert(bundle_id.clone());
                    }
                } else {
                    self.tool_call_bundle_states.insert(bundle_id.clone());
                    self.user_collapsed_bundles.remove(bundle_id);
                }
            }
            ToolCallDisplay::Expanded => {}
        }
    }

    pub(crate) fn tool_call_bundle_state(
        &self,
        bundle_id: &BundleId,
        tool_call_display: ToolCallDisplay,
    ) -> bool {
        match tool_call_display {
            ToolCallDisplay::Expanded => true,
            ToolCallDisplay::Auto | ToolCallDisplay::Compact => {
                self.tool_call_bundle_states.contains(bundle_id)
            }
        }
    }

    /// Returns true when the entry is a non-first member of a collapsed
    /// bundle — i.e. it renders as `Empty` and takes no visual space.
    pub(crate) fn is_entry_in_collapsed_bundle(
        &self,
        entry_ix: usize,
        tool_call_display: ToolCallDisplay,
    ) -> bool {
        match get_bundle_for_entry(&self.bundles, entry_ix) {
            Some(bundle) if entry_ix != bundle.start_index => {
                !self.tool_call_bundle_state(&bundle.id, tool_call_display)
            }
            _ => false,
        }
    }

    pub(crate) fn auto_expand_streaming_bundles(&mut self, thread: &AcpThread, cx: &App) -> bool {
        if AgentSettings::get_global(cx).tool_call_display != ToolCallDisplay::Auto {
            return false;
        }
        if thread.status() != ThreadStatus::Generating {
            return false;
        }
        let Some(latest) = self.bundles.last() else {
            return false;
        };
        let key = latest.id.clone();

        if self.tool_call_bundle_states.contains(&key) {
            return false;
        }
        if self.user_collapsed_bundles.contains(&key) {
            return false;
        }

        if let Some(old) = self.auto_expanded_bundle.take() {
            self.tool_call_bundle_states.remove(&old);
        }
        self.tool_call_bundle_states.insert(key.clone());
        self.auto_expanded_bundle = Some(key);
        true
    }

    pub(crate) fn auto_compact_bundles(&mut self, cx: &App) -> bool {
        if AgentSettings::get_global(cx).tool_call_display == ToolCallDisplay::Expanded {
            return false;
        }

        let mut changed = false;
        if let Some(key) = self.auto_expanded_bundle.take() {
            self.tool_call_bundle_states.remove(&key);
            changed = true;
        }
        if !self.user_collapsed_bundles.is_empty() {
            self.user_collapsed_bundles.clear();
            changed = true;
        }
        changed
    }

    pub(crate) fn recompute_bundles(
        &mut self,
        entries: &[AgentThreadEntry],
        hint: Option<BundleRecomputeHint>,
    ) {
        // Skip recompute for ToolCalls whose eligibility hasn't changed and
        // that aren't at a bundle boundary. AssistantMessages always
        // recompute because their thought chunk count may change during
        // streaming without affecting eligibility.
        if let Some(BundleRecomputeHint::EntryUpdated(index)) = &hint {
            let is_tool_call = matches!(entries.get(*index), Some(AgentThreadEntry::ToolCall(_)));
            if is_tool_call {
                let is_eligible = entries.get(*index).is_some_and(is_entry_bundle_eligible);
                let was_in_bundle = get_bundle_for_entry(&self.bundles, *index).is_some();
                let is_trailing = self
                    .bundles
                    .binary_search_by_key(index, |b| b.end_index)
                    .is_ok();

                if !is_trailing && was_in_bundle == is_eligible {
                    return;
                }
            }
        }

        let new_bundles = match &hint {
            Some(hint) => {
                let affected_start = self.affected_start(entries, hint);
                let prefix: Vec<ToolCallBundle> = self
                    .bundles
                    .iter()
                    .take_while(|b| b.end_index <= affected_start)
                    .cloned()
                    .collect();
                let suffix = compute_tool_call_bundles(&entries[affected_start..], affected_start);
                let mut all = prefix;
                all.extend(suffix);
                all
            }
            None => compute_tool_call_bundles(entries, 0),
        };
        self.bundles = Rc::from(new_bundles);
        self.prune_bundle_states();
    }

    /// Finds the earliest entry index whose bundle might be affected by the
    /// hinted change. Bundles entirely before this index are preserved as-is.
    fn affected_start(&self, entries: &[AgentThreadEntry], hint: &BundleRecomputeHint) -> usize {
        match hint {
            BundleRecomputeHint::NewEntry(index) | BundleRecomputeHint::EntryUpdated(index) => {
                let mut start = *index;
                if let Some(b) = get_bundle_for_entry(&self.bundles, *index) {
                    start = start.min(b.start_index);
                }
                if *index > 0 {
                    if let Some(b) = get_bundle_for_entry(&self.bundles, index - 1) {
                        if b.end_index == *index {
                            start = start.min(b.start_index);
                        }
                    }
                }
                // Walk back through contiguous eligible entries to catch
                // singletons that might form or join a bundle.
                while start > 0 && is_entry_bundle_eligible(&entries[start - 1]) {
                    start -= 1;
                }
                start
            }
            BundleRecomputeHint::EntriesRemoved { range } => range.start,
        }
    }

    /// Drops expansion state for bundles that no longer exist.
    fn prune_bundle_states(&mut self) {
        let valid_ids: HashSet<&BundleId> = self.bundles.iter().map(|b| &b.id).collect();
        self.tool_call_bundle_states
            .retain(|id| valid_ids.contains(id));
        self.user_collapsed_bundles
            .retain(|id| valid_ids.contains(id));
        if let Some(ref auto_id) = self.auto_expanded_bundle {
            if !valid_ids.contains(auto_id) {
                self.auto_expanded_bundle = None;
            }
        }
    }

    pub fn entry(&self, index: usize) -> Option<&Entry> {
        self.entries.get(index)
    }

    pub fn sync_entry(
        &mut self,
        index: usize,
        thread: &Entity<AcpThread>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(thread_entry) = thread.read(cx).entries().get(index) else {
            return;
        };

        match thread_entry {
            AgentThreadEntry::UserMessage(message) => {
                let can_rewind = thread.read(cx).supports_truncate(cx);
                let has_client_id = message.client_id.is_some();
                let is_subagent = thread.read(cx).parent_session_id().is_some();
                let chunks = message.chunks.clone();
                if let Some(Entry::UserMessage(editor)) = self.entries.get_mut(index) {
                    if !editor.focus_handle(cx).is_focused(window) {
                        // Only update if we are not editing.
                        // If we are, cancelling the edit will set the message to the newest content.
                        editor.update(cx, |editor, cx| {
                            editor.set_message(chunks, window, cx);
                        });
                    }
                } else {
                    let message_editor = cx.new(|cx| {
                        let mut editor = MessageEditor::new(
                            self.workspace.clone(),
                            self.project.clone(),
                            self.thread_store.clone(),
                            self.session_capabilities.clone(),
                            self.agent_id.clone(),
                            "Edit message － @ to include context",
                            editor::EditorMode::AutoHeight {
                                min_lines: 1,
                                max_lines: None,
                            },
                            window,
                            cx,
                        );
                        if !can_rewind || !has_client_id || is_subagent {
                            editor.set_read_only(true, cx);
                        }
                        editor.set_message(chunks, window, cx);
                        editor
                    });
                    cx.subscribe(&message_editor, move |_, editor, event, cx| {
                        cx.emit(EntryViewEvent {
                            entry_index: index,
                            view_event: ViewEvent::MessageEditorEvent(editor, event.clone()),
                        })
                    })
                    .detach();
                    self.set_entry(index, Entry::UserMessage(message_editor));
                }
            }
            AgentThreadEntry::ToolCall(tool_call) => {
                let id = tool_call.id.clone();
                let terminals = tool_call.terminals().cloned().collect::<Vec<_>>();
                let diffs = tool_call.diffs().cloned().collect::<Vec<_>>();

                let views = if let Some(Entry::ToolCall(tool_call)) = self.entries.get_mut(index) {
                    &mut tool_call.content
                } else {
                    self.set_entry(
                        index,
                        Entry::ToolCall(ToolCallEntry {
                            content: HashMap::default(),
                            focus_handle: cx.focus_handle(),
                        }),
                    );
                    let Some(Entry::ToolCall(tool_call)) = self.entries.get_mut(index) else {
                        unreachable!()
                    };
                    &mut tool_call.content
                };

                let is_tool_call_completed =
                    matches!(tool_call.status, acp_thread::ToolCallStatus::Completed);

                for terminal in terminals {
                    match views.entry(terminal.entity_id()) {
                        collections::hash_map::Entry::Vacant(entry) => {
                            let element = create_terminal(
                                self.workspace.clone(),
                                self.project.clone(),
                                terminal.clone(),
                                window,
                                cx,
                            )
                            .into_any();
                            cx.emit(EntryViewEvent {
                                entry_index: index,
                                view_event: ViewEvent::NewTerminal(id.clone()),
                            });
                            entry.insert(element);
                        }
                        collections::hash_map::Entry::Occupied(_entry) => {
                            if is_tool_call_completed && terminal.read(cx).output().is_none() {
                                cx.emit(EntryViewEvent {
                                    entry_index: index,
                                    view_event: ViewEvent::TerminalMovedToBackground(id.clone()),
                                });
                            }
                        }
                    }
                }

                for diff in diffs {
                    views.entry(diff.entity_id()).or_insert_with(|| {
                        let editor = create_editor_diff(diff.clone(), window, cx);
                        cx.subscribe(&editor, {
                            let diff = diff.clone();
                            let entry_index = index;
                            move |_this, _editor, event: &EditorEvent, cx| {
                                if let EditorEvent::OpenExcerptsRequested {
                                    selections_by_buffer,
                                    split,
                                } = event
                                {
                                    let multibuffer = diff.read(cx).multibuffer();
                                    if let Some((buffer_id, (ranges, _))) =
                                        selections_by_buffer.iter().next()
                                    {
                                        if let Some(buffer) =
                                            multibuffer.read(cx).buffer(*buffer_id)
                                        {
                                            if let Some(range) = ranges.first() {
                                                let point =
                                                    buffer.read(cx).offset_to_point(range.start.0);
                                                if let Some(path) = diff.read(cx).file_path(cx) {
                                                    cx.emit(EntryViewEvent {
                                                        entry_index,
                                                        view_event: ViewEvent::OpenDiffLocation {
                                                            path,
                                                            position: point,
                                                            split: *split,
                                                        },
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        })
                        .detach();
                        cx.emit(EntryViewEvent {
                            entry_index: index,
                            view_event: ViewEvent::NewDiff(id.clone()),
                        });
                        editor.into_any()
                    });
                }
            }
            AgentThreadEntry::AssistantMessage(message) => {
                let entry = if let Some(Entry::AssistantMessage(entry)) =
                    self.entries.get_mut(index)
                {
                    entry
                } else {
                    self.set_entry(
                        index,
                        Entry::AssistantMessage(AssistantMessageEntry {
                            scroll_handles_by_chunk_index: HashMap::default(),
                            focus_handle: cx.focus_handle(),
                        }),
                    );
                    let Some(Entry::AssistantMessage(entry)) = self.entries.get_mut(index) else {
                        unreachable!()
                    };
                    entry
                };
                entry.sync(message);
            }
            AgentThreadEntry::CompletedPlan(_) => {
                if !matches!(self.entries.get(index), Some(Entry::CompletedPlan)) {
                    self.set_entry(index, Entry::CompletedPlan);
                }
            }
            AgentThreadEntry::ContextCompaction(_) => {
                if !matches!(self.entries.get(index), Some(Entry::ContextCompaction)) {
                    self.set_entry(index, Entry::ContextCompaction);
                }
            }
        };
    }

    fn set_entry(&mut self, index: usize, entry: Entry) {
        if index == self.entries.len() {
            self.entries.push(entry);
        } else {
            self.entries[index] = entry;
        }
    }

    pub fn remove(&mut self, range: Range<usize>) {
        self.entries.drain(range.clone());

        self.expanded_compactions = self
            .expanded_compactions
            .iter()
            .filter_map(|&entry_ix| reindex_after_removal(entry_ix, &range))
            .collect();
        self.expanded_thinking_blocks = self
            .expanded_thinking_blocks
            .iter()
            .filter_map(|&(entry_ix, chunk_ix)| {
                reindex_after_removal(entry_ix, &range).map(|entry_ix| (entry_ix, chunk_ix))
            })
            .collect();
        self.user_toggled_thinking_blocks = self
            .user_toggled_thinking_blocks
            .iter()
            .filter_map(|&(entry_ix, chunk_ix)| {
                reindex_after_removal(entry_ix, &range).map(|entry_ix| (entry_ix, chunk_ix))
            })
            .collect();
        self.auto_expanded_thinking_block =
            self.auto_expanded_thinking_block
                .and_then(|(entry_ix, chunk_ix)| {
                    reindex_after_removal(entry_ix, &range).map(|entry_ix| (entry_ix, chunk_ix))
                });
    }

    pub fn agent_ui_font_size_changed(&mut self, cx: &mut App) {
        for entry in self.entries.iter() {
            match entry {
                Entry::UserMessage { .. }
                | Entry::AssistantMessage { .. }
                | Entry::CompletedPlan
                | Entry::ContextCompaction => {}
                Entry::ToolCall(ToolCallEntry { content, .. }) => {
                    for view in content.values() {
                        if let Ok(diff_editor) = view.clone().downcast::<Editor>() {
                            diff_editor.update(cx, |diff_editor, cx| {
                                diff_editor.set_text_style_refinement(
                                    diff_editor_text_style_refinement(cx),
                                );
                                cx.notify();
                            })
                        }
                    }
                }
            }
        }
    }
}

impl EventEmitter<EntryViewEvent> for EntryViewState {}

pub struct EntryViewEvent {
    pub entry_index: usize,
    pub view_event: ViewEvent,
}

pub enum ViewEvent {
    NewDiff(acp::ToolCallId),
    NewTerminal(acp::ToolCallId),
    TerminalMovedToBackground(acp::ToolCallId),
    MessageEditorEvent(Entity<MessageEditor>, MessageEditorEvent),
    OpenDiffLocation {
        path: String,
        position: Point,
        split: bool,
    },
}

#[derive(Debug)]
pub struct AssistantMessageEntry {
    scroll_handles_by_chunk_index: HashMap<usize, ScrollHandle>,
    focus_handle: FocusHandle,
}

impl AssistantMessageEntry {
    pub fn scroll_handle_for_chunk(&self, ix: usize) -> Option<ScrollHandle> {
        self.scroll_handles_by_chunk_index.get(&ix).cloned()
    }

    pub fn sync(&mut self, message: &acp_thread::AssistantMessage) {
        if let Some(acp_thread::AssistantMessageChunk::Thought { .. }) = message.chunks.last() {
            let ix = message.chunks.len() - 1;
            let handle = self.scroll_handles_by_chunk_index.entry(ix).or_default();
            handle.scroll_to_bottom();
        }
    }
}

#[derive(Debug)]
pub struct ToolCallEntry {
    content: HashMap<EntityId, AnyEntity>,
    focus_handle: FocusHandle,
}

#[derive(Debug)]
pub enum Entry {
    UserMessage(Entity<MessageEditor>),
    AssistantMessage(AssistantMessageEntry),
    ToolCall(ToolCallEntry),
    CompletedPlan,
    ContextCompaction,
}

impl Entry {
    pub fn focus_handle(&self, cx: &App) -> Option<FocusHandle> {
        match self {
            Self::UserMessage(editor) => Some(editor.read(cx).focus_handle(cx)),
            Self::AssistantMessage(message) => Some(message.focus_handle.clone()),
            Self::ToolCall(tool_call) => Some(tool_call.focus_handle.clone()),
            Self::CompletedPlan | Self::ContextCompaction => None,
        }
    }

    pub fn message_editor(&self) -> Option<&Entity<MessageEditor>> {
        match self {
            Self::UserMessage(editor) => Some(editor),
            Self::AssistantMessage(_)
            | Self::ToolCall(_)
            | Self::CompletedPlan
            | Self::ContextCompaction => None,
        }
    }

    pub fn editor_for_diff(&self, diff: &Entity<acp_thread::Diff>) -> Option<Entity<Editor>> {
        self.content_map()?
            .get(&diff.entity_id())
            .cloned()
            .map(|entity| entity.downcast::<Editor>().unwrap())
    }

    pub fn terminal(
        &self,
        terminal: &Entity<acp_thread::Terminal>,
    ) -> Option<Entity<TerminalView>> {
        self.content_map()?
            .get(&terminal.entity_id())
            .cloned()
            .map(|entity| entity.downcast::<TerminalView>().unwrap())
    }

    pub fn scroll_handle_for_assistant_message_chunk(
        &self,
        chunk_ix: usize,
    ) -> Option<ScrollHandle> {
        match self {
            Self::AssistantMessage(message) => message.scroll_handle_for_chunk(chunk_ix),
            Self::UserMessage(_)
            | Self::ToolCall(_)
            | Self::CompletedPlan
            | Self::ContextCompaction => None,
        }
    }

    fn content_map(&self) -> Option<&HashMap<EntityId, AnyEntity>> {
        match self {
            Self::ToolCall(ToolCallEntry { content, .. }) => Some(content),
            _ => None,
        }
    }

    #[cfg(test)]
    pub fn has_content(&self) -> bool {
        match self {
            Self::ToolCall(ToolCallEntry { content, .. }) => !content.is_empty(),
            Self::UserMessage(_)
            | Self::AssistantMessage(_)
            | Self::CompletedPlan
            | Self::ContextCompaction => false,
        }
    }
}

impl Focusable for ToolCallEntry {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Focusable for Entry {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        match self {
            Self::UserMessage(editor) => editor.read(cx).focus_handle(cx),
            Self::AssistantMessage(message) => message.focus_handle.clone(),
            Self::ToolCall(tool_call) => tool_call.focus_handle.clone(),
            Self::CompletedPlan | Self::ContextCompaction => cx.focus_handle(),
        }
    }
}

fn create_terminal(
    workspace: WeakEntity<Workspace>,
    project: WeakEntity<Project>,
    terminal: Entity<acp_thread::Terminal>,
    window: &mut Window,
    cx: &mut App,
) -> Entity<TerminalView> {
    cx.new(|cx| {
        let mut view = TerminalView::new(
            terminal.read(cx).inner().clone(),
            workspace,
            None,
            project,
            window,
            cx,
        );
        view.set_embedded_mode(Some(1000), cx);
        view
    })
}

fn create_editor_diff(
    diff: Entity<acp_thread::Diff>,
    window: &mut Window,
    cx: &mut App,
) -> Entity<Editor> {
    cx.new(|cx| {
        let mut editor = Editor::new(
            EditorMode::Full {
                scale_ui_elements_with_buffer_font_size: false,
                show_active_line_background: false,
                sizing_behavior: SizingBehavior::SizeByContent,
            },
            diff.read(cx).multibuffer().clone(),
            None,
            window,
            cx,
        );
        editor.set_show_gutter(false, cx);
        editor.disable_diagnostics(cx);
        editor.set_max_diagnostics_severity(DiagnosticSeverity::Off, cx);
        editor.disable_expand_excerpt_buttons(cx);
        editor.set_show_vertical_scrollbar(false, cx);
        editor.set_minimap_visibility(MinimapVisibility::Disabled, window, cx);
        editor.set_soft_wrap_mode(SoftWrap::None, cx);
        editor.set_forbid_vertical_scroll(true);
        editor.set_show_indent_guides(false, cx);
        editor.set_read_only(true);
        editor.set_delegate_open_excerpts(true);
        editor.set_show_bookmarks(false, cx);
        editor.set_show_breakpoints(false, cx);
        editor.set_show_code_actions(false, cx);
        editor.set_show_git_diff_gutter(false, cx);
        editor.set_expand_all_diff_hunks(cx);
        editor.set_render_diff_hunks_as_unstaged(true, cx);
        editor.set_text_style_refinement(diff_editor_text_style_refinement(cx));
        editor
    })
}

fn diff_editor_text_style_refinement(cx: &mut App) -> TextStyleRefinement {
    TextStyleRefinement {
        font_size: Some(
            TextSize::Small
                .rems(cx)
                .to_pixels(ThemeSettings::get_global(cx).agent_ui_font_size(cx))
                .into(),
        ),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::rc::Rc;
    use std::sync::Arc;

    use acp_thread::{AgentConnection, AgentThreadEntry, StubAgentConnection};
    use agent_client_protocol::schema::v1 as acp;
    use agent_settings;
    use buffer_diff::{DiffHunkStatus, DiffHunkStatusKind};
    use editor::RowInfo;
    use fs::FakeFs;
    use gpui::{AppContext as _, TestAppContext};
    use parking_lot::RwLock;

    use crate::entry_view_state::EntryViewState;
    use crate::message_editor::SessionCapabilities;
    use multi_buffer::MultiBufferRow;
    use pretty_assertions::assert_matches;
    use project::Project;
    use serde_json::json;
    use settings::{Settings, SettingsStore};
    use util::path;
    use workspace::{MultiWorkspace, PathList};

    #[test]
    fn test_reindex_after_removal() {
        use super::reindex_after_removal;

        // Entries before the removed range keep their index.
        assert_eq!(reindex_after_removal(0, &(2..4)), Some(0));
        assert_eq!(reindex_after_removal(1, &(2..4)), Some(1));
        // Entries inside the removed range are dropped.
        assert_eq!(reindex_after_removal(2, &(2..4)), None);
        assert_eq!(reindex_after_removal(3, &(2..4)), None);
        // Entries after the removed range slide down by its length.
        assert_eq!(reindex_after_removal(4, &(2..4)), Some(2));
        assert_eq!(reindex_after_removal(5, &(2..4)), Some(3));
        // An empty removal range leaves indices untouched.
        assert_eq!(reindex_after_removal(3, &(2..2)), Some(3));
    }

    #[gpui::test]
    async fn test_diff_sync(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                "hello.txt": "hi world"
            }),
        )
        .await;
        let project = Project::test(fs, [Path::new(path!("/project"))], cx).await;

        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |mw, _| mw.workspace().clone());

        let tool_call = acp::ToolCall::new("tool", "Tool call")
            .status(acp::ToolCallStatus::InProgress)
            .content(vec![acp::ToolCallContent::Diff(
                acp::Diff::new("/project/hello.txt", "hello world").old_text("hi world"),
            )]);
        let connection = Rc::new(StubAgentConnection::new());
        let thread = cx
            .update(|_, cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[Path::new(path!("/project"))]),
                    cx,
                )
            })
            .await
            .unwrap();
        let session_id = thread.update(cx, |thread, _| thread.session_id().clone());

        cx.update(|_, cx| {
            connection.send_update(session_id, acp::SessionUpdate::ToolCall(tool_call), cx)
        });

        let thread_store = None;

        let view_state = cx.new(|_cx| {
            EntryViewState::new(
                workspace.downgrade(),
                project.downgrade(),
                thread_store,
                Arc::new(RwLock::new(SessionCapabilities::default())),
                "Test Agent".into(),
            )
        });

        view_state.update_in(cx, |view_state, window, cx| {
            view_state.sync_entry(0, &thread, window, cx)
        });

        let diff = thread.read_with(cx, |thread, _| {
            thread
                .entries()
                .get(0)
                .unwrap()
                .diffs()
                .next()
                .unwrap()
                .clone()
        });

        cx.run_until_parked();

        let diff_editor = view_state.read_with(cx, |view_state, _cx| {
            view_state.entry(0).unwrap().editor_for_diff(&diff).unwrap()
        });
        assert_eq!(
            diff_editor.read_with(cx, |editor, cx| editor.text(cx)),
            "hi world\nhello world"
        );
        let row_infos = diff_editor.read_with(cx, |editor, cx| {
            let multibuffer = editor.buffer().read(cx);
            multibuffer
                .snapshot(cx)
                .row_infos(MultiBufferRow(0))
                .collect::<Vec<_>>()
        });
        assert_matches!(
            row_infos.as_slice(),
            [
                RowInfo {
                    multibuffer_row: Some(MultiBufferRow(0)),
                    diff_status: Some(DiffHunkStatus {
                        kind: DiffHunkStatusKind::Deleted,
                        ..
                    }),
                    ..
                },
                RowInfo {
                    multibuffer_row: Some(MultiBufferRow(1)),
                    diff_status: Some(DiffHunkStatus {
                        kind: DiffHunkStatusKind::Added,
                        ..
                    }),
                    ..
                }
            ]
        );
    }

    // --- Helpers for bundle tests ---

    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TOOL_CALL_ID: AtomicU64 = AtomicU64::new(0);

    fn unique_tool_call_id() -> acp::ToolCallId {
        acp::ToolCallId::new(format!(
            "tool-call-{}",
            NEXT_TOOL_CALL_ID.fetch_add(1, Ordering::SeqCst)
        ))
    }

    fn tool_call_entry(
        cx: &mut gpui::App,
        kind: acp::ToolKind,
        status: acp_thread::ToolCallStatus,
    ) -> AgentThreadEntry {
        use acp_thread::ToolCall;
        AgentThreadEntry::ToolCall(ToolCall {
            id: unique_tool_call_id(),
            label: cx.new(|cx| markdown::Markdown::new("test".into(), None, None, cx)),
            kind,
            content: vec![],
            status,
            locations: vec![],
            resolved_locations: vec![],
            raw_input: None,
            raw_input_markdown: None,
            raw_output: None,
            tool_name: None,
            subagent_session_info: None,
            sandbox_authorization_details: None,
            sandbox_fallback_authorization_details: None,
            sandbox_not_applied: None,
        })
    }

    fn tool_call_with_tool_name(
        cx: &mut gpui::App,
        kind: acp::ToolKind,
        status: acp_thread::ToolCallStatus,
        tool_name: impl Into<gpui::SharedString>,
    ) -> AgentThreadEntry {
        use acp_thread::ToolCall;
        AgentThreadEntry::ToolCall(ToolCall {
            id: unique_tool_call_id(),
            label: cx.new(|cx| markdown::Markdown::new("test".into(), None, None, cx)),
            kind,
            content: vec![],
            status,
            locations: vec![],
            resolved_locations: vec![],
            raw_input: None,
            raw_input_markdown: None,
            raw_output: None,
            tool_name: Some(tool_name.into()),
            subagent_session_info: None,
            sandbox_authorization_details: None,
            sandbox_fallback_authorization_details: None,
            sandbox_not_applied: None,
        })
    }

    fn thought_entry() -> AgentThreadEntry {
        use acp_thread::{AssistantMessage, AssistantMessageChunk, ContentBlock};
        AgentThreadEntry::AssistantMessage(AssistantMessage {
            chunks: vec![AssistantMessageChunk::Thought {
                id: None,
                block: ContentBlock::Empty,
            }],
            indented: false,
            is_subagent_output: false,
        })
    }

    fn user_entry() -> AgentThreadEntry {
        use acp_thread::{ContentBlock, UserMessage};
        AgentThreadEntry::UserMessage(UserMessage {
            protocol_id: None,
            client_id: None,
            is_optimistic: false,
            content: ContentBlock::Empty,
            chunks: vec![],
            checkpoint: None,
            indented: false,
        })
    }

    /// A mixed AssistantMessage with the given number of thought chunks
    /// plus one non-thought (message) chunk. Not bundle-eligible.
    fn mixed_thought_message(thought_count: usize) -> AgentThreadEntry {
        use acp_thread::{AssistantMessage, AssistantMessageChunk, ContentBlock};
        let mut chunks: Vec<_> = (0..thought_count)
            .map(|_| AssistantMessageChunk::Thought {
                id: None,
                block: ContentBlock::Empty,
            })
            .collect();
        chunks.push(AssistantMessageChunk::Message {
            id: None,
            block: ContentBlock::Empty,
        });
        AgentThreadEntry::AssistantMessage(AssistantMessage {
            chunks,
            indented: false,
            is_subagent_output: false,
        })
    }

    // --- Bundle computation tests ---

    fn kind_count(bundle: &super::ToolCallBundle, kind: acp::ToolKind) -> u32 {
        bundle
            .kind_counts_iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, c)| c)
            .unwrap_or(0)
    }

    #[gpui::test]
    async fn test_compute_tool_call_bundles_empty(cx: &mut TestAppContext) {
        init_test(cx);
        let entries = cx.update(|_| vec![]);
        let bundles = super::compute_tool_call_bundles(&entries, 0);
        assert!(bundles.is_empty());
    }

    #[gpui::test]
    async fn test_compute_tool_call_bundles_two_tool_calls(cx: &mut TestAppContext) {
        init_test(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        let bundles = super::compute_tool_call_bundles(&entries, 0);
        assert_eq!(bundles.len(), 1);
        let bundle = &bundles[0];
        assert_eq!(bundle.start_index, 0);
        assert_eq!(bundle.end_index, 2);
        assert_eq!(bundle.tool_call_count, 2);
        assert_eq!(bundle.thought_count, 0);
        assert_eq!(kind_count(bundle, acp::ToolKind::Read), 1);
        assert_eq!(kind_count(bundle, acp::ToolKind::Edit), 1);
    }

    #[gpui::test]
    async fn test_compute_tool_call_bundles_mixed_tool_calls_and_thoughts(cx: &mut TestAppContext) {
        init_test(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                thought_entry(),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
                thought_entry(),
            ]
        });
        let bundles = super::compute_tool_call_bundles(&entries, 0);
        assert_eq!(bundles.len(), 1);
        let bundle = &bundles[0];
        assert_eq!(bundle.start_index, 0);
        assert_eq!(bundle.end_index, 4);
        assert_eq!(bundle.tool_call_count, 2);
        assert_eq!(bundle.thought_count, 2);
        assert_eq!(kind_count(bundle, acp::ToolKind::Read), 1);
        assert_eq!(kind_count(bundle, acp::ToolKind::Edit), 1);
    }

    #[gpui::test]
    async fn test_compute_tool_call_bundles_boundary_at_user_message(cx: &mut TestAppContext) {
        init_test(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
                user_entry(),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        let bundles = super::compute_tool_call_bundles(&entries, 0);
        assert_eq!(bundles.len(), 2);
        assert_eq!(bundles[0].start_index, 0);
        assert_eq!(bundles[0].end_index, 2);
        assert_eq!(bundles[1].start_index, 3);
        assert_eq!(bundles[1].end_index, 5);
    }

    #[gpui::test]
    async fn test_compute_tool_call_bundles_boundary_at_waiting_for_confirmation(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let (respond_tx, _respond_rx) = futures::channel::oneshot::channel();
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::WaitingForConfirmation {
                        current_status: acp::ToolCallStatus::Pending,
                        options: acp_thread::PermissionOptions::Flat(vec![]),
                        respond_tx,
                        kind: acp_thread::AuthorizationKind::PermissionGrant,
                    },
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        let bundles = super::compute_tool_call_bundles(&entries, 0);
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].start_index, 1);
        assert_eq!(bundles[0].end_index, 3);
        assert_eq!(bundles[0].tool_call_count, 2);
    }

    #[gpui::test]
    async fn test_compute_tool_call_bundles_subagent_not_eligible(cx: &mut TestAppContext) {
        init_test(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_with_tool_name(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                    "spawn_agent",
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        let bundles = super::compute_tool_call_bundles(&entries, 0);
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].start_index, 1);
        assert_eq!(bundles[0].end_index, 3);
    }

    #[gpui::test]
    async fn test_compute_tool_call_bundles_pure_thought_message(cx: &mut TestAppContext) {
        init_test(cx);
        let entries = cx.update(|_| vec![thought_entry(), thought_entry()]);
        let bundles = super::compute_tool_call_bundles(&entries, 0);
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].start_index, 0);
        assert_eq!(bundles[0].end_index, 2);
        assert_eq!(bundles[0].tool_call_count, 0);
        assert_eq!(bundles[0].thought_count, 2);
    }

    #[gpui::test]
    async fn test_compute_tool_call_bundles_mixed_message_terminates(cx: &mut TestAppContext) {
        init_test(cx);
        // A mixed message (thoughts + text) is eligible but terminating —
        // it ends the bundle. [Tool, Mixed, Tool, Tool] => two bundles.
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                mixed_thought_message(1),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        let bundles = super::compute_tool_call_bundles(&entries, 0);
        assert_eq!(bundles.len(), 2);
        assert_eq!(bundles[0].start_index, 0);
        assert_eq!(bundles[0].end_index, 2);
        assert_eq!(bundles[0].tool_call_count, 1);
        assert_eq!(bundles[0].thought_count, 1);
        assert_eq!(bundles[1].start_index, 2);
        assert_eq!(bundles[1].end_index, 4);
        assert_eq!(bundles[1].tool_call_count, 2);
    }

    #[gpui::test]
    async fn test_compute_tool_call_bundles_pure_text_message_is_boundary(cx: &mut TestAppContext) {
        init_test(cx);
        // A pure-text message (no thoughts) is a hard boundary.
        use acp_thread::{AssistantMessage, AssistantMessageChunk, ContentBlock};
        let text_message = AgentThreadEntry::AssistantMessage(AssistantMessage {
            chunks: vec![AssistantMessageChunk::Message {
                id: None,
                block: ContentBlock::Empty,
            }],
            indented: false,
            is_subagent_output: false,
        });
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                text_message,
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        let bundles = super::compute_tool_call_bundles(&entries, 0);
        // The text message breaks the run: two singletons, no bundle.
        assert!(bundles.is_empty());
    }

    // --- get_bundle_for_entry tests ---

    #[gpui::test]
    async fn test_get_bundle_for_entry_in_bundle(cx: &mut TestAppContext) {
        init_test(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        let bundles = super::compute_tool_call_bundles(&entries, 0);
        assert_eq!(bundles.len(), 1);

        let found = super::get_bundle_for_entry(&bundles, 0);
        assert!(found.is_some());
        assert_eq!(found.unwrap().start_index, 0);

        let found = super::get_bundle_for_entry(&bundles, 1);
        assert!(found.is_some());
    }

    #[gpui::test]
    async fn test_get_bundle_for_entry_not_in_bundle(cx: &mut TestAppContext) {
        init_test(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
                user_entry(),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        let bundles = super::compute_tool_call_bundles(&entries, 0);
        assert_eq!(bundles.len(), 2);

        assert!(super::get_bundle_for_entry(&bundles, 2).is_none());
    }

    fn init_test(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
    }

    /// Creates an `EntryViewState` with invalid weak entity references.
    fn make_entry_view_state(cx: &mut TestAppContext) -> gpui::Entity<EntryViewState> {
        cx.new(|_cx| {
            EntryViewState::new(
                gpui::WeakEntity::new_invalid(),
                gpui::WeakEntity::new_invalid(),
                None,
                Arc::new(parking_lot::RwLock::new(SessionCapabilities::default())),
                project::AgentId::new("Test Agent"),
            )
        })
    }

    // ---------------------------------------------------------------------------
    // recompute_bundles tests
    // ---------------------------------------------------------------------------

    #[gpui::test]
    async fn test_recompute_new_entry_appends_to_run(cx: &mut TestAppContext) {
        init_test(cx);
        let view_state = make_entry_view_state(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&entries, None);
        });
        assert_eq!(view_state.read_with(cx, |s, _| s.bundles().len()), 1);

        // Append Tool(Delete, Completed) at index 2
        let mut updated = entries;
        updated.push(cx.update(|cx| {
            tool_call_entry(
                cx,
                acp::ToolKind::Delete,
                acp_thread::ToolCallStatus::Completed,
            )
        }));
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&updated, Some(super::BundleRecomputeHint::NewEntry(2)));
        });

        let bundle = view_state.read_with(cx, |s, _| s.bundles()[0].clone());
        assert_eq!(bundle.start_index, 0);
        assert_eq!(bundle.end_index, 3);
        assert_eq!(bundle.tool_call_count, 3);
    }

    #[gpui::test]
    async fn test_recompute_new_entry_ineligible(cx: &mut TestAppContext) {
        init_test(cx);
        let view_state = make_entry_view_state(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&entries, None);
        });

        // Append a user message at index 2 (ineligible)
        let mut updated = entries;
        updated.push(user_entry());
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&updated, Some(super::BundleRecomputeHint::NewEntry(2)));
        });

        let bundles = view_state.read_with(cx, |s, _| s.bundles().clone());
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].start_index, 0);
        assert_eq!(bundles[0].end_index, 2);
    }

    #[gpui::test]
    async fn test_recompute_entry_updated_becomes_ineligible(cx: &mut TestAppContext) {
        init_test(cx);
        let view_state = make_entry_view_state(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Delete,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&entries, None);
        });

        // Entry at index 1 becomes Canceled (ineligible)
        let mut updated = entries;
        updated[1] = cx.update(|cx| {
            tool_call_entry(
                cx,
                acp::ToolKind::Edit,
                acp_thread::ToolCallStatus::Canceled,
            )
        });
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&updated, Some(super::BundleRecomputeHint::EntryUpdated(1)));
        });

        let bundles = view_state.read_with(cx, |s, _| s.bundles().clone());
        assert!(
            bundles.is_empty(),
            "waiting tool breaks eligibility; no bundles expected"
        );
    }

    #[gpui::test]
    async fn test_recompute_entry_updated_becomes_eligible(cx: &mut TestAppContext) {
        init_test(cx);
        let view_state = make_entry_view_state(cx);
        let (respond_tx, _respond_rx) = futures::channel::oneshot::channel();
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::WaitingForConfirmation {
                        current_status: acp::ToolCallStatus::Pending,
                        options: acp_thread::PermissionOptions::Flat(vec![]),
                        respond_tx,
                        kind: acp_thread::AuthorizationKind::PermissionGrant,
                    },
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Delete,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&entries, None);
        });
        assert!(
            view_state.read_with(cx, |s, _| s.bundles().is_empty()),
            "waiting tool breaks eligibility; no bundles expected"
        );

        // Entry at index 1 becomes Completed (now eligible)
        let mut updated = entries;
        updated[1] = cx.update(|cx| {
            tool_call_entry(
                cx,
                acp::ToolKind::Edit,
                acp_thread::ToolCallStatus::Completed,
            )
        });
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&updated, Some(super::BundleRecomputeHint::EntryUpdated(1)));
        });

        let bundles = view_state.read_with(cx, |s, _| s.bundles().clone());
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].start_index, 0);
        assert_eq!(bundles[0].end_index, 3);
        assert_eq!(bundles[0].tool_call_count, 3);
    }

    #[gpui::test]
    async fn test_recompute_entry_updated_no_change_skips(cx: &mut TestAppContext) {
        init_test(cx);
        let view_state = make_entry_view_state(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&entries, None);
        });
        let original = view_state.read_with(cx, |s, _| s.bundles().clone());

        // "Update" entry 0 to the same state (still eligible)
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&entries, Some(super::BundleRecomputeHint::EntryUpdated(0)));
        });

        // Bundles unchanged (same Rc pointer — recompute was skipped)
        assert!(std::ptr::eq(
            view_state
                .read_with(cx, |s, _| s.bundles().clone())
                .as_ref(),
            original.as_ref()
        ));
    }

    #[gpui::test]
    async fn test_recompute_split_run_handled_correctly(cx: &mut TestAppContext) {
        init_test(cx);
        let view_state = make_entry_view_state(cx);
        let entries = cx.update(|cx| {
            (0..10)
                .map(|_| {
                    tool_call_entry(
                        cx,
                        acp::ToolKind::Read,
                        acp_thread::ToolCallStatus::Completed,
                    )
                })
                .collect::<Vec<_>>()
        });
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&entries, None);
        });

        // Make entry 5 Canceled (ineligible)
        let mut updated = entries;
        updated[5] = cx.update(|cx| {
            tool_call_entry(
                cx,
                acp::ToolKind::Read,
                acp_thread::ToolCallStatus::Canceled,
            )
        });
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&updated, Some(super::BundleRecomputeHint::EntryUpdated(5)));
        });

        let bundles = view_state.read_with(cx, |s, _| s.bundles().clone());
        assert_eq!(bundles.len(), 2);
        assert_eq!(bundles[0].start_index, 0);
        assert_eq!(bundles[0].end_index, 5);
        assert_eq!(bundles[0].tool_call_count, 5);
        assert_eq!(bundles[1].start_index, 6);
        assert_eq!(bundles[1].end_index, 10);
        assert_eq!(bundles[1].tool_call_count, 4);
    }

    #[gpui::test]
    async fn test_recompute_entries_removed(cx: &mut TestAppContext) {
        init_test(cx);
        let view_state = make_entry_view_state(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
                user_entry(),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Delete,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Execute,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&entries, None);
        });
        assert_eq!(view_state.read_with(cx, |s, _| s.bundles().len()), 2);

        // Remove entries 3..5 (the last two tool calls)
        let reduced: Vec<_> = entries.into_iter().take(3).collect();
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(
                &reduced,
                Some(super::BundleRecomputeHint::EntriesRemoved { range: 3..5 }),
            );
        });

        let bundles = view_state.read_with(cx, |s, _| s.bundles().clone());
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].start_index, 0);
        assert_eq!(bundles[0].end_index, 2);
    }

    // ---------------------------------------------------------------------------
    // State management tests
    // ---------------------------------------------------------------------------

    #[gpui::test]
    async fn test_toggle_bundle_expansion(cx: &mut TestAppContext) {
        init_test(cx);
        let view_state = make_entry_view_state(cx);
        let bundle_id = super::BundleId::ToolCall(acp::ToolCallId::new("test-tc"));

        // Default: not expanded
        assert!(!view_state.read_with(cx, |s, _| {
            s.tool_call_bundle_state(&bundle_id, settings::ToolCallDisplay::Auto)
        }));

        // Toggle on
        view_state.update(cx, |state, cx| {
            state.toggle_tool_call_bundle_expansion(&bundle_id, cx);
        });
        assert!(view_state.read_with(cx, |s, _| {
            s.tool_call_bundle_state(&bundle_id, settings::ToolCallDisplay::Auto)
        }));

        // Toggle off
        view_state.update(cx, |state, cx| {
            state.toggle_tool_call_bundle_expansion(&bundle_id, cx);
        });
        assert!(!view_state.read_with(cx, |s, _| {
            s.tool_call_bundle_state(&bundle_id, settings::ToolCallDisplay::Auto)
        }));

        // ToolCallDisplay::Expanded always returns true
        view_state.update(cx, |_, cx| {
            agent_settings::AgentSettings::override_global(
                agent_settings::AgentSettings {
                    tool_call_display: settings::ToolCallDisplay::Expanded,
                    ..agent_settings::AgentSettings::get_global(cx).clone()
                },
                cx,
            );
        });
        assert!(view_state.read_with(cx, |s, _| {
            s.tool_call_bundle_state(&bundle_id, settings::ToolCallDisplay::Expanded)
        }));
    }

    #[gpui::test]
    async fn test_bundle_state_survives_removal(cx: &mut TestAppContext) {
        init_test(cx);
        let view_state = make_entry_view_state(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
                user_entry(),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Delete,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Execute,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&entries, None);
        });
        let bundles = view_state.read_with(cx, |s, _| s.bundles().clone());
        assert_eq!(bundles.len(), 2);

        // Expand the first bundle (user action)
        let first_id = bundles[0].id.clone();
        view_state.update(cx, |state, cx| {
            state.toggle_tool_call_bundle_expansion(&first_id, cx);
        });

        // Remove the second bundle (entries 3..5)
        let reduced: Vec<_> = entries.into_iter().take(3).collect();
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(
                &reduced,
                Some(super::BundleRecomputeHint::EntriesRemoved { range: 3..5 }),
            );
        });

        // First bundle's expansion state survives because it's keyed by
        // ToolCallId, not positional index.
        assert!(view_state.read_with(cx, |s, _| {
            s.tool_call_bundle_state(&first_id, settings::ToolCallDisplay::Auto)
        }));
    }

    #[gpui::test]
    async fn test_auto_compact_bundles(cx: &mut TestAppContext) {
        init_test(cx);
        let view_state = make_entry_view_state(cx);
        let bundle_id = super::BundleId::ToolCall(acp::ToolCallId::new("auto-tc"));

        // Simulate auto-expand by inserting into the set and setting the field.
        view_state.update(cx, |state, _cx| {
            state.tool_call_bundle_states.insert(bundle_id.clone());
            state.auto_expanded_bundle = Some(bundle_id.clone());
        });

        // auto_compact should remove the auto-expanded bundle
        let changed = view_state.update(cx, |state, cx| state.auto_compact_bundles(cx));
        assert!(changed);

        assert!(!view_state.read_with(cx, |s, _| {
            s.tool_call_bundle_state(&bundle_id, settings::ToolCallDisplay::Auto)
        }));

        // Calling auto_compact again with nothing to compact returns false
        let changed = view_state.update(cx, |state, cx| state.auto_compact_bundles(cx));
        assert!(!changed);

        // auto_compact should NOT remove a user-toggled bundle
        let user_id = super::BundleId::ToolCall(acp::ToolCallId::new("user-tc"));
        view_state.update(cx, |state, cx| {
            state.toggle_tool_call_bundle_expansion(&user_id, cx);
        });

        let changed = view_state.update(cx, |state, cx| state.auto_compact_bundles(cx));
        assert!(!changed);

        assert!(view_state.read_with(cx, |s, _| {
            s.tool_call_bundle_state(&user_id, settings::ToolCallDisplay::Auto)
        }));
    }

    #[gpui::test]
    async fn test_user_collapse_resists_auto_expand(cx: &mut TestAppContext) {
        init_test(cx);
        let view_state = make_entry_view_state(cx);
        let entries = cx.update(|cx| {
            vec![
                tool_call_entry(
                    cx,
                    acp::ToolKind::Read,
                    acp_thread::ToolCallStatus::Completed,
                ),
                tool_call_entry(
                    cx,
                    acp::ToolKind::Edit,
                    acp_thread::ToolCallStatus::Completed,
                ),
            ]
        });
        view_state.update(cx, |state, _cx| {
            state.recompute_bundles(&entries, None);
        });

        let bundle_id = view_state.read_with(cx, |s, _| s.bundles()[0].id.clone());

        // Simulate auto-expand (set ToolCallDisplay::Auto + Generating status
        // is hard in unit tests, so we call auto_expand directly on the state).
        // First, mark the thread as generating by inserting auto state manually.
        view_state.update(cx, |state, _cx| {
            state.tool_call_bundle_states.insert(bundle_id.clone());
            state.auto_expanded_bundle = Some(bundle_id.clone());
        });

        // User collapses the auto-expanded bundle
        view_state.update(cx, |state, cx| {
            state.toggle_tool_call_bundle_expansion(&bundle_id, cx);
        });

        // The bundle should be collapsed
        assert!(!view_state.read_with(cx, |s, _| {
            s.tool_call_bundle_state(&bundle_id, settings::ToolCallDisplay::Auto)
        }));

        // And recorded in user_collapsed_bundles to prevent re-auto-expand
        assert!(view_state.read_with(cx, |s, _| { s.user_collapsed_bundles.contains(&bundle_id) }));

        // auto_compact clears user_collapsed_bundles for the next turn
        view_state.update(cx, |state, cx| {
            state.auto_compact_bundles(cx);
        });
        assert!(
            !view_state.read_with(cx, |s, _| { s.user_collapsed_bundles.contains(&bundle_id) })
        );
    }
}

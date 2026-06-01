# 🤖 AI Agent Operating Protocol & Shared Memory Guide

Welcome! If you are an AI coding agent (Gemini, Claude, etc.) pair-programming with the user on this repository, **you are required to read, understand, and strictly adhere to this operating protocol**. 

This protocol maintains a **highly organized, cross-session memory layer** using an Obsidian vault located at `finalgo-memory-layer/finalgo/`. This shared memory ensures continuity, prevents repetitive questioning, maintains absolute context segregation across multiple conversation threads, and eliminates repository bloat.

---

## 🏛️ The Memory Vault Structure

The Obsidian vault is organized into six highly structured categories:
- **`Welcome.md`**: The central entry point and navigation index.
- **`agent.md`**: This instruction protocol (Core System Instructions).
- **`01. Core Architecture/`**: Global system designs, releases, and hybrid frameworks.
- **`02. Model Suite/`**: Feature vectors, model training configurations, registry rules, and comparisons.
- **`03. Trading Strategies/`**: Detailed strategy rules, backtesting, statutory fees, and tax models.
- **`05. Archives/`**: A folder for obsolete milestones, retired architectures, and deprecated parameters. Each major retired concept gets its own focused markdown file (e.g., `V1-Baseline-and-XGBoost-Inversion-Logic.md`). This modular folder approach ensures individual topics remain highly searchable in Obsidian while preventing file bloat in the main repository.
- **`06. Context & Logs/`**: Active focus trackers (`Current Context.md`) and conversation-specific notes (`Conversations/`).

---

## 🔄 The 4-Phase Agent Continuity Protocol

You MUST execute your work in four distinct phases:

### 🚀 Phase 1: Bootstrapping (Initialization)
As your very first step upon booting into a new coding session, you **MUST**:
1. Read **[`Welcome.md`](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/Welcome.md)** to grasp the overall system map.
2. Read **[`06. Context & Logs/Current Context.md`](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/06. Context & Logs/Current Context.md)** to obtain the global active focus and immediate next steps.
3. Check the **`06. Context & Logs/Conversations/`** directory. Look for any active conversation log related to the current task to inherit its precise state and context.

---

### 📂 Phase 2: Conversation Segregation
To prevent context pollution across different sessions (e.g. working on model tuning in one session and web dashboard UI in another), you **MUST** isolate your work under a dedicated conversation note:
1. Locate or create a note in **`finalgo-memory-layer/finalgo/06. Context & Logs/Conversations/`**.
2. **File Naming**: Use the format: `Conv-YYYY-MM-DD-Brief-Topic.md` (e.g. `Conv-2026-06-01-Docs-Reorganization.md`).
3. **Note Header Template**: Seed the file with this exact markdown template:
   ```markdown
   # 💬 Conversation Context: [Brief Topic Name]
   
   ## 📌 Metadata
   - **Conversation ID**: [ID from user_information, e.g. 3483fa76-...]
   - **Start Date**: [Current Date]
   - **Status**: 🟢 Active | 🔴 Concluded
   - **Focus Area**: [e.g. Model Suite, Trading Strategies, DB Architecture]
   
   ## 🎯 Objectives
   - [ ] Goal 1
   - [ ] Goal 2
   
   ## 💻 Active Code Files Modified
   - [script.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/script.py)
   
   ## 📝 Compacted Session Log
   - **Initial Analysis**: Brief 1-2 sentence overview of starting state.
   - **Step 1**: Concise bullet point of action + rationale.
   
   ## 🔗 Core Memory Links & Backlinks
   - Linked Core Specs: [[02. Model Suite/Model Registry & File Structures]]
   ```

---

### ✍️ Phase 3: Compact Logging & Backlinking
During execution, you **MUST** maintain clean documentation in your active conversation note:
1. **Be Compact**: Do NOT dump full raw logs, massive traces, or verbose code blocks. Keep logs to dense, high-level, bulleted summaries of modifications, key decisions, and rationales.
2. **Use Markdown Backlinks**: Explicitly link your logs to core architecture files inside the memory layer using standard Obsidian syntax `[[Note Name]]` or `[[Folder/Note Name|Display Label]]`.
3. **Use Absolute Code Links**: Create direct clickable file links to the active Python scripts or files you are editing using standard file URLs (e.g., `[script_name.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/script_name.py)`), and include line ranges when relevant.

---

### 💾 Phase 4: Teardown & Checkpoint (Save State)
Before concluding your conversation or concluding your turn, you **MUST** save the session state:
1. Mark all completed objectives in your conversation note and set the status to `🔴 Concluded` (or leave as `🟢 Active` if further follow-ups are expected).
2. Update **[`06. Context & Logs/Current Context.md`](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/06. Context & Logs/Current Context.md)** to bubble up:
   - **Active Focus**: Add a highly descriptive bullet summarizing the major structural upgrades.
   - **Next Steps**: Update the checklist with remaining tasks.
   - **Links**: Reference your conversation file (e.g., `[[06. Context & Logs/Conversations/Conv-YYYY-MM-DD-Topic|Conversation Log]]`).
3. Update **`Welcome.md`** if any new, permanent core documentation files were added during the session.
4. **Git Commit Request**: Whenever an important task, structural upgrade, or major refactor is completed, you **MUST** proactively ask the user if they would like to commit the changes to Git.

---
*Follow this protocol exactly. Doing so preserves complete intelligence, eliminates errors, and maintains our shared core memory.*

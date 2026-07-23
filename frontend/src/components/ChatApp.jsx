import { useEffect, useState } from "react";
import { api } from "../api.js";
import Sidebar from "./Sidebar.jsx";
import Conversation from "./Conversation.jsx";

// Merge a patch into the last message of the list (used to stream tokens and
// attach sources to the in-flight assistant reply). `patch` may be an object or
// a function of the current message.
function patchLast(messages, patch) {
  if (messages.length === 0) return messages;
  const i = messages.length - 1;
  const last = messages[i];
  const next = typeof patch === "function" ? patch(last) : patch;
  return [...messages.slice(0, i), { ...last, ...next }];
}

// Owns the per-user state (conversations + messages + documents) and the
// actions that mutate it, handing both down to the sidebar and the conversation
// pane. Everything here is already scoped to the signed-in user by the token the
// API sends.
//
// A "new chat" is lazy: selecting it just clears the pane (activeId = null). The
// conversation is created server-side on the first question (or first document
// upload), so empty chats never clutter the list.
//
// Documents are scoped to a single chat: each conversation has its own uploaded
// documents, and a new chat starts empty — nothing ingested in other chats
// carries over. The documents list therefore always reflects the active chat.
export default function ChatApp({ user, onLogout }) {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [sending, setSending] = useState(false);
  const [banner, setBanner] = useState("");

  useEffect(() => {
    bootstrap();
  }, []);

  async function bootstrap() {
    try {
      const convs = await api.conversations();
      setConversations(convs);
      if (convs.length > 0) {
        await openConversation(convs[0].id);
      } else {
        startNewChat();
      }
    } catch {
      startNewChat();
    }
  }

  function refreshConversations() {
    return api.conversations().then(setConversations).catch(() => {});
  }

  // Documents belong to a chat, so fetching is scoped to a conversation id. A
  // new chat (no id yet) has none, so we just clear the list.
  function refreshDocuments(conversationId) {
    if (!conversationId) {
      setDocuments([]);
      return Promise.resolve();
    }
    return api.documents(conversationId).then(setDocuments).catch(() => {});
  }

  function flash(msg) {
    setBanner(msg);
    setTimeout(() => setBanner(""), 4000);
  }

  async function openConversation(id) {
    setActiveId(id);
    refreshDocuments(id);
    try {
      const msgs = await api.conversationMessages(id);
      setMessages(msgs);
    } catch {
      setMessages([]);
    }
  }

  function startNewChat() {
    setActiveId(null);
    setMessages([]);
    setDocuments([]);
  }

  async function ask(question) {
    const q = question.trim();
    if (!q || sending) return;
    setSending(true);
    // Push the user turn and an empty assistant bubble we stream tokens into.
    setMessages((m) => [
      ...m,
      { role: "user", content: q },
      { role: "assistant", content: "", sources: [], streaming: true },
    ]);
    try {
      await api.queryStream(
        { question: q, conversation_id: activeId },
        {
          onMeta: (meta) => {
            // The server may have created the conversation on this first message.
            if (meta.conversation_id !== activeId)
              setActiveId(meta.conversation_id);
            setMessages((m) => patchLast(m, { sources: meta.sources }));
          },
          onToken: (text) => {
            setMessages((m) =>
              patchLast(m, (msg) => ({ content: msg.content + text }))
            );
          },
        }
      );
      setMessages((m) => patchLast(m, { streaming: false }));
      refreshConversations();
    } catch (err) {
      setMessages((m) =>
        patchLast(m, { content: `⚠ ${err.message}`, error: true, streaming: false })
      );
    } finally {
      setSending(false);
    }
  }

  async function deleteConversation(id) {
    await api.deleteConversation(id).catch(() => {});
    const remaining = conversations.filter((c) => c.id !== id);
    setConversations(remaining);
    if (id === activeId) {
      if (remaining.length > 0) await openConversation(remaining[0].id);
      else startNewChat();
    }
  }

  async function upload(file) {
    try {
      const job = await api.uploadDocument(file, activeId);
      // Uploading into a brand-new chat creates the conversation server-side;
      // adopt its id so this document (and the rest of the chat) stays bound to it.
      const convId = job.conversation_id;
      if (convId && convId !== activeId) {
        setActiveId(convId);
        refreshConversations();
      }
      flash(`Indexing “${job.source}”…`);
      const final = await pollIngest(job.job_id);
      if (final.status === "done") {
        flash(`Added “${final.source}” — ${final.chunks_added} chunks indexed.`);
        await refreshDocuments(convId || activeId);
      } else {
        flash(final.detail || "Ingestion failed.");
      }
    } catch (err) {
      flash(err.message);
    }
  }

  async function pollIngest(jobId) {
    for (;;) {
      const job = await api.ingestStatus(jobId);
      if (job.status === "done" || job.status === "error") return job;
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  async function deleteAccount() {
    const password = window.prompt(
      "Permanently delete your account, documents, and chats?\n" +
        "This cannot be undone. Enter your password to confirm:"
    );
    if (!password) return;
    try {
      await api.deleteAccount(password);
      onLogout();
    } catch (err) {
      flash(err.message);
    }
  }

  async function removeDocument(source) {
    if (!activeId) return;
    await api.deleteDocument(source, activeId).catch(() => {});
    await refreshDocuments(activeId);
  }

  return (
    <div className="shell">
      <Sidebar
        user={user}
        conversations={conversations}
        activeId={activeId}
        onSelectConversation={openConversation}
        onNewChat={startNewChat}
        onDeleteConversation={deleteConversation}
        documents={documents}
        onUpload={upload}
        onDelete={removeDocument}
        onLogout={onLogout}
        onDeleteAccount={deleteAccount}
      />
      <Conversation
        messages={messages}
        sending={sending}
        banner={banner}
        hasDocuments={documents.length > 0}
        onAsk={ask}
      />
    </div>
  );
}

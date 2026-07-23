import { useRef, useState } from "react";

// Left rail: identity, the user's saved chats (selectable history), the
// documents uploaded into the *active* chat (upload / remove), and account
// actions. Everything shown here belongs to this user alone, and the document
// shelf is scoped to the current chat — each chat has its own documents.
export default function Sidebar({
  user,
  conversations,
  activeId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  documents,
  onUpload,
  onDelete,
  onLogout,
  onDeleteAccount,
}) {
  const fileRef = useRef(null);
  const [uploading, setUploading] = useState(false);

  async function pick(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    await onUpload(file);
    setUploading(false);
    e.target.value = "";
  }

  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="brandmark">§</span>
        <span className="brandname">Vault</span>
      </div>

      <div className="sidebar__who">
        <span className="who__name">{user.username}</span>
        <span className="who__email">{user.email}</span>
      </div>

      <div className="chats">
        <div className="shelf__head">
          <h2>Chats</h2>
          <span className="shelf__count">{conversations.length}</span>
        </div>

        <button className="btn btn--ghost btn--block" onClick={onNewChat}>
          + New chat
        </button>

        <ul className="convos">
          {conversations.length === 0 && (
            <li className="convos__empty">
              No saved chats yet. Ask a question to start one.
            </li>
          )}
          {conversations.map((c) => (
            <li
              key={c.id}
              className={`convo ${c.id === activeId ? "convo--active" : ""}`}
            >
              <button
                className="convo__open"
                title={c.title}
                onClick={() => onSelectConversation(c.id)}
              >
                <span className="convo__title">{c.title}</span>
              </button>
              <button
                className="convo__remove"
                title="Delete chat"
                aria-label={`Delete ${c.title}`}
                onClick={() => onDeleteConversation(c.id)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="shelf">
        <div className="shelf__head">
          <h2>Chat documents</h2>
          <span className="shelf__count">{documents.length}</span>
        </div>

        <button
          className="btn btn--ghost btn--block"
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? "Indexing…" : "+ Add document"}
        </button>
        {/* No `accept` filter: text, code, PDF, Word, Excel, PowerPoint,
            images, and .zip are all supported — the server validates the type
            and rejects anything it can't read with a clear message. */}
        <input ref={fileRef} type="file" hidden onChange={pick} />

        <ul className="docs">
          {documents.length === 0 && (
            <li className="docs__empty">
              Nothing in this chat yet. Add a PDF, text, or markdown file to
              start asking questions here.
            </li>
          )}
          {documents.map((d) => (
            <li className="doc" key={d.source}>
              <div className="doc__body">
                <span className="doc__name" title={d.source}>
                  {d.source}
                </span>
                <span className="doc__meta">{d.chunks} chunks</span>
              </div>
              <button
                className="doc__remove"
                title="Remove document"
                aria-label={`Remove ${d.source}`}
                onClick={() => onDelete(d.source)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="sidebar__foot">
        <button className="btn btn--quiet btn--block" onClick={onLogout}>
          Sign out
        </button>
        <button
          className="btn btn--quiet btn--block btn--danger"
          onClick={onDeleteAccount}
        >
          Delete account
        </button>
      </div>
    </aside>
  );
}

import { useEffect, useRef, useState } from "react";

// Right pane: the running conversation plus the composer. Each assistant answer
// can carry its retrieved chunks, which render as archive "call cards" — the
// signature element, since citing the source is the whole point of RAG.
export default function Conversation({
  messages,
  sending,
  banner,
  hasDocuments,
  onAsk,
}) {
  const [draft, setDraft] = useState("");
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  function submit(e) {
    e.preventDefault();
    onAsk(draft);
    setDraft("");
  }

  return (
    <main className="conversation">
      <header className="conversation__top">
        <h1>Reading room</h1>
        <p>Answers are drawn only from documents on your shelf.</p>
      </header>

      {banner && <div className="banner">{banner}</div>}

      <div className="thread">
        {messages.length === 0 && <EmptyState hasDocuments={hasDocuments} />}
        {messages.map((m, i) => (
          <Bubble key={i} message={m} />
        ))}
        <div ref={endRef} />
      </div>

      <form className="composer" onSubmit={submit}>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit(e);
            }
          }}
          placeholder="Ask something about your documents…"
          rows={1}
        />
        <button className="btn btn--primary" disabled={sending || !draft.trim()}>
          Ask
        </button>
      </form>
    </main>
  );
}

function Bubble({ message }) {
  const isUser = message.role === "user";
  // An assistant bubble that's streaming but hasn't produced text yet shows the
  // typing indicator; once tokens arrive it renders them live.
  const waiting = message.streaming && !message.content;
  return (
    <div className={`bubble ${isUser ? "bubble--user" : "bubble--assistant"}`}>
      <div className="bubble__role">{isUser ? "You" : "Vault"}</div>
      {waiting ? (
        <div className="bubble__body typing">
          <span></span>
          <span></span>
          <span></span>
        </div>
      ) : (
        <div className={`bubble__body ${message.error ? "bubble__body--error" : ""}`}>
          {message.content}
        </div>
      )}
      {message.sources && message.sources.length > 0 && (
        <Sources sources={message.sources} />
      )}
    </div>
  );
}

function Sources({ sources }) {
  return (
    <div className="sources">
      {/* "Retrieved", not "Drawn from": these are the chunks search returned and
          handed to the model. When they don't cover the question the model says
          so and answers from general knowledge instead, so claiming the answer
          came from them would be false. Citations in the answer say what it used. */}
      <div className="sources__label">Retrieved from your documents</div>
      <div className="sources__cards">
        {sources.map((s, i) => (
          <article className="callcard" key={i}>
            <header className="callcard__head">
              <span className="callcard__name" title={s.source}>
                {s.source}
              </span>
              <Score value={s.score} />
            </header>
            <p className="callcard__excerpt">{s.text}</p>
          </article>
        ))}
      </div>
    </div>
  );
}

function Score({ value }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <span className="score" title={`relevance ${value.toFixed(3)}`}>
      <span className="score__bar">
        <span className="score__fill" style={{ width: `${pct}%` }} />
      </span>
      <span className="score__num">{value.toFixed(2)}</span>
    </span>
  );
}

function EmptyState({ hasDocuments }) {
  return (
    <div className="empty">
      <span className="empty__mark">§</span>
      {hasDocuments ? (
        <>
          <h2>Ask away.</h2>
          <p>
            Your shelf is stocked. Ask a question and the answer will cite the
            documents it came from.
          </p>
        </>
      ) : (
        <>
          <h2>Start by adding a document.</h2>
          <p>
            Use “Add document” on the left to upload a PDF, text, or markdown
            file. Once it’s indexed, ask anything about it here.
          </p>
        </>
      )}
    </div>
  );
}

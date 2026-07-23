import { useEffect, useState } from "react";
import { api } from "../api.js";

const MODES = {
  login: "Sign in",
  signup: "Create account",
  forgot: "Reset password",
};

export default function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [form, setForm] = useState({});
  // forgot flow: once we've fetched the question, show step 2
  const [question, setQuestion] = useState(null);
  // preset security questions, fetched from the server for the signup dropdown
  const [securityQuestions, setSecurityQuestions] = useState([]);

  useEffect(() => {
    api
      .securityQuestions()
      .then((res) => setSecurityQuestions(res.security_questions))
      .catch(() => {});
  }, []);

  function set(key) {
    return (e) => setForm((f) => ({ ...f, [key]: e.target.value }));
  }

  function switchMode(next) {
    setMode(next);
    setError("");
    setQuestion(null);
  }

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (mode === "login") {
        onAuthenticated(
          await api.login({
            identifier: form.identifier,
            password: form.password,
          })
        );
      } else if (mode === "signup") {
        onAuthenticated(
          await api.signup({
            username: form.username,
            email: form.email,
            password: form.password,
            security_question: form.security_question,
            security_answer: form.security_answer,
          })
        );
      } else if (mode === "forgot") {
        if (!question) {
          const res = await api.forgot({ identifier: form.identifier });
          setQuestion(res.security_question);
        } else {
          onAuthenticated(
            await api.reset({
              identifier: form.identifier,
              security_answer: form.security_answer,
              new_password: form.new_password,
            })
          );
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth">
      <div className="auth__aside">
        <div className="auth__brand">
          <span className="brandmark">§</span>
          <span className="brandname">Vault</span>
        </div>
        <h1 className="auth__pitch">
          A reading room
          <br />
          that is only yours.
        </h1>
        <p className="auth__sub">
          Upload your documents, ask questions in plain language, and get
          answers drawn only from your own shelf — with every source cited.
          Nobody else can see what you keep here.
        </p>
        <ul className="auth__notes">
          <li>Private vector store, scoped to your account</li>
          <li>Fresh chat each time you sign in</li>
          <li>Answers cite the page they came from</li>
        </ul>
      </div>

      <div className="auth__panel">
        <div className="auth__tabs">
          {Object.entries(MODES).map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={`tab ${mode === key ? "tab--on" : ""}`}
              onClick={() => switchMode(key)}
            >
              {label}
            </button>
          ))}
        </div>

        <form className="auth__form" onSubmit={submit}>
          {mode === "signup" && (
            <>
              <Field label="Username">
                <input required minLength={3} value={form.username || ""} onChange={set("username")} />
              </Field>
              <Field label="Email">
                <input required type="email" value={form.email || ""} onChange={set("email")} />
              </Field>
            </>
          )}

          {(mode === "login" || mode === "forgot") && (
            <Field label="Username or email">
              <input
                required
                value={form.identifier || ""}
                onChange={set("identifier")}
                disabled={mode === "forgot" && question}
              />
            </Field>
          )}

          {mode === "login" && (
            <Field label="Password">
              <input required type="password" value={form.password || ""} onChange={set("password")} />
            </Field>
          )}

          {mode === "signup" && (
            <>
              <Field label="Password" hint="At least 8 characters">
                <input required type="password" minLength={8} value={form.password || ""} onChange={set("password")} />
              </Field>
              <Field label="Security question" hint="Used to recover your account">
                <select required value={form.security_question || ""} onChange={set("security_question")}>
                  <option value="" disabled>
                    Choose a question…
                  </option>
                  {securityQuestions.map((q) => (
                    <option key={q} value={q}>
                      {q}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Security answer">
                <input required value={form.security_answer || ""} onChange={set("security_answer")} />
              </Field>
            </>
          )}

          {mode === "forgot" && question && (
            <>
              <Field label="Security question">
                <input value={question} disabled />
              </Field>
              <Field label="Your answer">
                <input required value={form.security_answer || ""} onChange={set("security_answer")} />
              </Field>
              <Field label="New password" hint="At least 8 characters">
                <input required type="password" minLength={8} value={form.new_password || ""} onChange={set("new_password")} />
              </Field>
            </>
          )}

          {error && <p className="auth__error">{error}</p>}

          <button className="btn btn--primary" disabled={busy}>
            {busy
              ? "Working…"
              : mode === "forgot" && !question
              ? "Find my account"
              : MODES[mode]}
          </button>
        </form>
      </div>
    </div>
  );
}

function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="field__label">
        {label}
        {hint && <span className="field__hint">{hint}</span>}
      </span>
      {children}
    </label>
  );
}

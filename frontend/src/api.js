// Tiny fetch wrapper. The JWT is kept in localStorage and attached to every
// request; a 401 means the session is gone, which the auth layer reacts to.

const TOKEN_KEY = "vault.token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(path, { method = "GET", body, json = true } = {}) {
  const headers = {};
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  let payload = body;
  if (json && body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }

  const res = await fetch(`/api${path}`, { method, headers, body: payload });

  if (res.status === 204) return null;

  let data = null;
  const text = await res.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!res.ok) {
    const detail =
      (data && data.detail) || `Request failed (${res.status}).`;
    throw new ApiError(
      Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail,
      res.status
    );
  }
  return data;
}

// Stream a chat answer via Server-Sent Events. Calls onMeta({conversation_id,
// title, sources}) once, then onToken(text) for each chunk. Resolves when the
// stream completes; rejects on transport or server-emitted errors.
async function queryStream(body, { onMeta, onToken } = {}) {
  const headers = { "Content-Type": "application/json" };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch("/api/chat/query/stream", {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    throw new ApiError(`Request failed (${res.status}).`, res.status);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handle = (event, data) => {
    if (event === "meta") onMeta?.(data);
    else if (event === "token") onToken?.(data.text);
    else if (event === "error") throw new ApiError(data.detail || "Stream error.");
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = "message";
      let dataStr = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
      }
      if (dataStr) handle(event, JSON.parse(dataStr));
    }
  }
}

export const api = {
  // auth
  signup: (b) => request("/auth/signup", { method: "POST", body: b }),
  login: (b) => request("/auth/login", { method: "POST", body: b }),
  forgot: (b) => request("/auth/forgot", { method: "POST", body: b }),
  reset: (b) => request("/auth/reset", { method: "POST", body: b }),
  securityQuestions: () => request("/auth/security-questions"),
  me: () => request("/auth/me"),
  logout: () => request("/auth/logout", { method: "POST" }),
  deleteAccount: (password) =>
    request("/auth/me", { method: "DELETE", body: { password } }),

  // chat
  query: (b) => request("/chat/query", { method: "POST", body: b }),
  queryStream,

  // conversations (selectable chats)
  conversations: () => request("/chat/conversations"),
  createConversation: (title) =>
    request("/chat/conversations", { method: "POST", body: { title } }),
  conversationMessages: (id) =>
    request(`/chat/conversations/${encodeURIComponent(id)}/messages`),
  deleteConversation: (id) =>
    request(`/chat/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),

  // documents (scoped to a single chat / conversation)
  documents: (conversationId) =>
    request(
      `/documents?conversation_id=${encodeURIComponent(conversationId)}`
    ),
  deleteDocument: (source, conversationId) =>
    request(
      `/documents/${encodeURIComponent(source)}?conversation_id=${encodeURIComponent(
        conversationId
      )}`,
      { method: "DELETE" }
    ),
  uploadDocument: (file, conversationId) => {
    const fd = new FormData();
    fd.append("file", file);
    // A brand-new chat has no id yet; the server creates the conversation and
    // returns its id on the job so we can bind subsequent turns to it.
    if (conversationId) fd.append("conversation_id", conversationId);
    // Returns a job ({ job_id, status, conversation_id, ... }); ingestion runs
    // in the background.
    return request("/documents", { method: "POST", body: fd, json: false });
  },
  ingestStatus: (jobId) =>
    request(`/documents/jobs/${encodeURIComponent(jobId)}`),
};

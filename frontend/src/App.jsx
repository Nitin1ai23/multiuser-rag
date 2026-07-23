import { useEffect, useState } from "react";
import { api, getToken, setToken } from "./api.js";
import AuthScreen from "./components/AuthScreen.jsx";
import ChatApp from "./components/ChatApp.jsx";

// Top-level: decide whether we have a valid session, then route between the
// auth screen and the app. On load, an existing token is validated via /me so
// a stale or forged token drops the user cleanly back to sign-in.
export default function App() {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    window.__vaultBoot?.step(0.75); // mounted; the session check is all that's left
    if (!getToken()) {
      setReady(true);
      return;
    }
    api
      .me()
      .then(setUser)
      .catch(() => setToken(null))
      .finally(() => setReady(true));
  }, []);

  // Hand the loading screen its ending. This runs after React has painted the
  // screen below, so the dust scatters off the real UI rather than off a blank
  // page. The overlay owns its own exit and removes itself.
  useEffect(() => {
    if (ready) window.__vaultBoot?.finish();
  }, [ready]);

  function onAuthenticated({ access_token, user }) {
    setToken(access_token);
    setUser(user);
  }

  function logout() {
    // Revoke the token server-side, then clear it locally regardless of result.
    api.logout().catch(() => {});
    setToken(null);
    setUser(null);
  }

  // Nothing to render while the session is checked: the loading overlay from
  // index.html is still up and covering the page.
  if (!ready) return null;

  return user ? (
    <ChatApp user={user} onLogout={logout} />
  ) : (
    <AuthScreen onAuthenticated={onAuthenticated} />
  );
}

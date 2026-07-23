import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

// The loading overlay in index.html is already up and reporting progress; this
// is the first thing it can't know on its own — the bundle arrived and ran.
window.__vaultBoot?.step(0.5);

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

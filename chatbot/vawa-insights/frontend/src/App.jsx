import React, { useMemo, useState } from "react";

/**
 * V1 UI goals:
 * - Simple, fast chat
 * - Clear response sections:
 *   Direct Answer / Evidence / Interpretation / Caveats / Citations
 * - Minimal styling, easy to modify later
 */

function getApiBase() {
  // You can override in a `.env` file:
  // VITE_API_BASE=http://127.0.0.1:8000
  return import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
}

async function sendChat(apiBase, message, history) {
  const res = await fetch(`${apiBase}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Backend error (${res.status}): ${text}`);
  }
  return res.json();
}

function Section({ title, children }) {
  return (
    <div className="section">
      <div className="sectionTitle">{title}</div>
      <div className="sectionBody">{children}</div>
    </div>
  );
}

function CitationList({ citations }) {
  if (!citations || citations.length === 0) return <div>No citations returned.</div>;

  return (
    <ul className="list">
      {citations.map((c, idx) => (
        <li key={idx} className="listItem">
          <div className="mono">
            {c.citation_type || "citation"}{" "}
            {c.citation_id ? `(${c.citation_id})` : ""}
          </div>
          {c.title ? <div>{c.title}</div> : null}
          <div className="muted small mono">{JSON.stringify(c)}</div>
        </li>
      ))}
    </ul>
  );
}

export default function App() {
  const apiBase = useMemo(() => getApiBase(), []);

  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Ask me about 2021–2024 trends (sample data) or VAWA/methodology (sample knowledge base). Try: “Compare California and Texas in firearm involvement from 2021 to 2024.”",
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastResponse, setLastResponse] = useState(null);
  const [error, setError] = useState("");

  async function onSubmit(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setError("");
    setLoading(true);
    setLastResponse(null);

    const nextMessages = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);
    setInput("");

    try {
      // Backend accepts optional history; we pass a minimal history format.
      const history = nextMessages.map((m) => ({ role: m.role, content: m.content }));
      const json = await sendChat(apiBase, text, history);
      setLastResponse(json);
      setMessages((prev) => [...prev, { role: "assistant", content: json?.answer?.direct_answer || "(no direct answer)" }]);
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  const answer = lastResponse?.answer;
  const debug = lastResponse?.debug;

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">VAWA Insights Bot (V1 prototype)</div>
          <div className="subtitle">
            Frontend → FastAPI tools + lightweight document retrieval → structured answer with citations
          </div>
        </div>
        <div className="muted small">
          API: <span className="mono">{apiBase}</span>
        </div>
      </header>

      <main className="grid">
        <div className="card">
          <div className="cardTitle">Chat</div>
          <div className="chatWindow" aria-label="Chat messages">
            {messages.map((m, idx) => (
              <div key={idx} className={`msg ${m.role === "user" ? "msgUser" : "msgAssistant"}`}>
                <div className="msgRole">{m.role}</div>
                <div className="msgContent">{m.content}</div>
              </div>
            ))}
            {loading ? <div className="muted small">Thinking…</div> : null}
          </div>

          <form className="composer" onSubmit={onSubmit}>
            <input
              className="input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your question…"
            />
            <button className="button" type="submit" disabled={loading}>
              Send
            </button>
          </form>

          {error ? <div className="error">Error: {error}</div> : null}
        </div>

        <div className="card">
          <div className="cardTitle">Latest Response (structured)</div>

          {!answer ? (
            <div className="muted">
              Ask a question to see the structured response sections here.
            </div>
          ) : (
            <>
              <Section title="Direct Answer">
                <div style={{ whiteSpace: "pre-wrap" }}>{answer.direct_answer}</div>
              </Section>

              <Section title="Evidence">
                {answer.evidence?.length ? (
                  <ul className="list">
                    {answer.evidence.map((x, i) => (
                      <li key={i} className="listItem mono">
                        {x}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="muted">No evidence lines returned.</div>
                )}
              </Section>

              <Section title="Interpretation">
                <div style={{ whiteSpace: "pre-wrap" }}>{answer.interpretation}</div>
              </Section>

              <Section title="Caveats">
                {answer.caveats?.length ? (
                  <ul className="list">
                    {answer.caveats.map((x, i) => (
                      <li key={i} className="listItem">
                        {x}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="muted">No caveats returned.</div>
                )}
              </Section>

              <Section title="Citations">
                <CitationList citations={answer.citations} />
              </Section>

              <Section title="Debug (V1)">
                <div className="muted small mono" style={{ whiteSpace: "pre-wrap" }}>
                  {JSON.stringify(debug, null, 2)}
                </div>
              </Section>
            </>
          )}
        </div>
      </main>

      <footer className="footer muted small">
        V1 note: this uses a small sample dataset + sample knowledge docs to prove the architecture.
      </footer>
    </div>
  );
}


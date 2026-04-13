import React, { useMemo, useState } from "react";

/**
 * Victim Tool Bot UI (V1 prototype)
 *
 * UI goals:
 * - Simple and calm chat experience
 * - Visible prototype + sample data warnings
 * - Render response sections exactly:
 *   Support Message / Resources / Practical Next Steps / Caveats / Citations
 */

function getApiBase() {
  // Optional override in `.env`:
  // VITE_API_BASE=http://127.0.0.1:8001
  return import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001";
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
      <div>{children}</div>
    </div>
  );
}

function ResourceCard({ r }) {
  return (
    <div className="listItem">
      <div style={{ fontWeight: 700 }}>{r.name}</div>
      <div className="muted small">
        <span className="mono">{r.resource_type}</span>
        {r.location ? <span> • {r.location}</span> : null}
      </div>
      {r.phone ? (
        <div>
          <span className="muted small">Phone: </span>
          <span className="mono">{r.phone}</span>
        </div>
      ) : null}
      {r.website ? (
        <div>
          <span className="muted small">Website: </span>
          <a href={r.website} target="_blank" rel="noreferrer">
            {r.website}
          </a>
        </div>
      ) : null}
      {r.notes ? <div className="muted small">{r.notes}</div> : null}
      {r.source ? <div className="muted small">Source: {r.source}</div> : null}
    </div>
  );
}

export default function App() {
  const apiBase = useMemo(() => getApiBase(), []);

  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi. I’m a prototype support tool. If you’re in immediate danger, call 911 (U.S.) or your local emergency number.\n\nYou can ask:\n- “I need help near Albuquerque.”\n- “What protections did VAWA add for dating partners?”\n- “I am scared and I don’t know what to do.”",
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
      const history = nextMessages.map((m) => ({ role: m.role, content: m.content }));
      const json = await sendChat(apiBase, text, history);
      setLastResponse(json);

      const support = json?.response?.support_message || "(no support_message returned)";
      setMessages((prev) => [...prev, { role: "assistant", content: support }]);
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  const resp = lastResponse?.response;
  const debug = lastResponse?.debug;
  const intent = debug?.intent;
  const isCrisis = intent === "crisis";

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">Victim Tool Bot (V1 prototype)</div>
          <div className="subtitle">
            Support + resource lookup + plain-language VAWA info. Built for reliability and safe wording (not therapy, not legal advice, not emergency
            services).
          </div>
        </div>
        <div className="muted small">
          API: <span className="mono">{apiBase}</span>
        </div>
      </header>

      <div className="bannerRow">
        <div className="banner">
          <strong>Prototype / demo</strong>: This is a local development prototype. Do not rely on it for real-time safety decisions.
        </div>
        <div className="banner">
          <strong>Sample resources notice</strong>: Some “local” listings are labeled <span className="mono">DEMO DATA</span> and are not verified services.
          The bot will not invent resources beyond what’s in its dataset.
        </div>
        <div className={`banner ${isCrisis ? "bannerDanger" : ""}`}>
          <strong>If you are in immediate danger</strong>: Call 911 (U.S.) or your local emergency number.
        </div>
      </div>

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
            <input className="input" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Type your message…" />
            <button className="button" type="submit" disabled={loading}>
              Send
            </button>
          </form>

          {error ? <div className="error">Error: {error}</div> : null}
        </div>

        <div className="card">
          <div className="cardTitle">Latest Response (structured)</div>

          {!resp ? (
            <div className="muted">Ask a question to see the structured response here.</div>
          ) : (
            <>
              <Section title="Support Message">
                <div style={{ whiteSpace: "pre-wrap" }}>{resp.support_message}</div>
              </Section>

              <Section title="Resources">
                {resp.resources?.length ? (
                  <div className="list">
                    {resp.resources.map((r, i) => (
                      <ResourceCard key={i} r={r} />
                    ))}
                  </div>
                ) : (
                  <div className="muted small">No resources returned.</div>
                )}
              </Section>

              <Section title="Practical Next Steps">
                {resp.practical_next_steps?.length ? (
                  <ul className="list">
                    {resp.practical_next_steps.map((x, i) => (
                      <li key={i} className="listItem">
                        {x}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="muted small">No next steps returned.</div>
                )}
              </Section>

              <Section title="Caveats">
                {resp.caveats?.length ? (
                  <ul className="list">
                    {resp.caveats.map((x, i) => (
                      <li key={i} className="listItem">
                        {x}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="muted small">No caveats returned.</div>
                )}
              </Section>

              <Section title="Citations">
                {resp.citations?.length ? (
                  <ul className="list">
                    {resp.citations.map((c, i) => (
                      <li key={i} className="listItem">
                        <span className="mono">{c}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="muted small">No citations returned.</div>
                )}
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
        V1 note: deterministic intent + sample resources + local KB retrieval. No geolocation APIs yet (ZIP/city/state matching only).
      </footer>
    </div>
  );
}


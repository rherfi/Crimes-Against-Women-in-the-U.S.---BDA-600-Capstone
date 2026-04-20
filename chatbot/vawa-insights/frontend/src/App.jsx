import React, { useMemo, useState } from "react";

function getApiBase() {
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

function SourcesAndLogicPanel({ answer, debug }) {
  if (!answer) return null;

  return (
    <div className="sourcesPanel" role="region" aria-label="Sources and reasoning">
      <Section title="Direct answer (full)">
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

      <Section title="Logic & retrieval (debug)">
        <div className="muted small mono" style={{ whiteSpace: "pre-wrap" }}>
          {JSON.stringify(debug, null, 2)}
        </div>
      </Section>
    </div>
  );
}

export default function App() {
  const apiBase = useMemo(() => getApiBase(), []);

  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Ask me about 2021–2024 trends (sample data) or VAWA/methodology (sample knowledge base). Try: “Compare California and Texas in firearm involvement from 2021 to 2024.”",
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [openDetailId, setOpenDetailId] = useState(null);

  async function onSubmit(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setError("");
    setLoading(true);
    setOpenDetailId(null);

    const userId = crypto.randomUUID();
    const nextMessages = [...messages, { id: userId, role: "user", content: text }];
    setMessages(nextMessages);
    setInput("");

    try {
      const history = nextMessages.map((m) => ({ role: m.role, content: m.content }));
      const json = await sendChat(apiBase, text, history);
      const assistantId = crypto.randomUUID();
      const direct = json?.answer?.direct_answer || "(no direct answer)";
      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: "assistant",
          content: direct,
          responseDetail: { answer: json?.answer, debug: json?.debug },
        },
      ]);
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  function toggleDetail(id) {
    setOpenDetailId((cur) => (cur === id ? null : id));
  }

  return (
    <div className="page">
      <header className="header">
        <div className="title">VAWA Insights Chatbot</div>
      </header>

      <main className="mainSingle">
        <div className="card cardChat">
          <div className="cardTitle">Chat</div>
          <div className="chatWindow" aria-label="Chat messages">
            {messages.map((m) => (
              <div key={m.id} className={`msg ${m.role === "user" ? "msgUser" : "msgAssistant"}`}>
                <div className="msgRole">{m.role}</div>
                <div className="msgContent">{m.content}</div>
                {m.role === "assistant" && m.responseDetail ? (
                  <div className="msgActions">
                    <button
                      type="button"
                      className="button buttonSecondary"
                      onClick={() => toggleDetail(m.id)}
                      aria-expanded={openDetailId === m.id}
                    >
                      {openDetailId === m.id ? "Hide sources & logic" : "Show sources & logic"}
                    </button>
                    {openDetailId === m.id ? (
                      <SourcesAndLogicPanel
                        answer={m.responseDetail.answer}
                        debug={m.responseDetail.debug}
                      />
                    ) : null}
                  </div>
                ) : null}
              </div>
            ))}
            {loading ? <div className="muted small chatThinking">Thinking…</div> : null}
          </div>

          <form className="composer" onSubmit={onSubmit}>
            <input
              className="input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your question…"
              aria-label="Message"
            />
            <button className="button" type="submit" disabled={loading}>
              Send
            </button>
          </form>

          {error ? <div className="error">Error: {error}</div> : null}
        </div>
      </main>
    </div>
  );
}

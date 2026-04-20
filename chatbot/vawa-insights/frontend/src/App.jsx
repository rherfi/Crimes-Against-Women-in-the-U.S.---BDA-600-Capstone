import React, { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const QUICK_EXIT_URL = "https://www.google.com/";

function getApiBase() {
  return import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
}

function quickExit() {
  window.location.replace(QUICK_EXIT_URL);
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

/**
 * POST /api/chat/stream (SSE). Falls back to /api/chat if stream is unavailable.
 */
async function sendChatStream(apiBase, message, history, handlers) {
  const streamRes = await fetch(`${apiBase}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ message, history }),
  });

  const ct = streamRes.headers.get("content-type") || "";
  if (!streamRes.ok && streamRes.status === 404) {
    handlers.onStart?.();
    const json = await sendChat(apiBase, message, history);
    const da = json?.answer?.direct_answer || "";
    const step = 24;
    for (let i = 0; i < da.length; i += step) {
      handlers.onDelta?.(da.slice(i, i + step));
    }
    handlers.onDone?.({ answer: json.answer, debug: json.debug });
    return;
  }

  if (!streamRes.ok) {
    const text = await streamRes.text();
    throw new Error(`Backend error (${streamRes.status}): ${text}`);
  }

  if (!streamRes.body || !ct.includes("text/event-stream")) {
    handlers.onStart?.();
    const json = await sendChat(apiBase, message, history);
    const da = json?.answer?.direct_answer || "";
    for (let i = 0; i < da.length; i += 24) {
      handlers.onDelta?.(da.slice(i, i + 24));
    }
    handlers.onDone?.({ answer: json.answer, debug: json.debug });
    return;
  }

  const reader = streamRes.body.getReader();
  const dec = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += dec.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of block.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        let data;
        try {
          data = JSON.parse(line.slice(6));
        } catch {
          continue;
        }
        if (data.type === "start") handlers.onStart?.();
        else if (data.type === "delta") handlers.onDelta?.(data.text || "");
        else if (data.type === "done") handlers.onDone?.(data);
        else if (data.type === "error") throw new Error(data.message || "Stream error");
      }
    }
  }
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

const markdownComponents = {
  a: (props) => <a {...props} target="_blank" rel="noopener noreferrer" />,
};

function MessageBody({ role, content, streaming }) {
  if (role === "user") {
    return <div className="msgContent msgPlain">{content}</div>;
  }
  if (streaming) {
    return <div className="msgContent msgPlain">{content}</div>;
  }
  return (
    <div className="msgContent msgMarkdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {content || ""}
      </ReactMarkdown>
    </div>
  );
}

function SourcesAndLogicPanel({ answer, debug }) {
  if (!answer) return null;

  return (
    <div className="sourcesPanel" role="region" aria-label="Sources and reasoning">
      <Section title="Direct answer (full)">
        <div className="msgMarkdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {answer.direct_answer || ""}
          </ReactMarkdown>
        </div>
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
        <div className="msgMarkdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {answer.interpretation || ""}
          </ReactMarkdown>
        </div>
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
      content: `Ask me about trends in violent crimes, VAWA policies, or resources for victims.

Example: *Compare California and Texas in firearm involvement from 2021 to 2024.*`,
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [openDetailId, setOpenDetailId] = useState(null);

  useEffect(() => {
    function onKeyDown(e) {
      if (e.key !== "Escape") return;
      if (openDetailId) {
        setOpenDetailId(null);
        e.preventDefault();
        return;
      }
      quickExit();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [openDetailId]);

  async function onSubmit(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setError("");
    setLoading(true);
    setOpenDetailId(null);

    const userId = crypto.randomUUID();
    const assistantId = crypto.randomUUID();
    const nextMessages = [...messages, { id: userId, role: "user", content: text }];
    setMessages([
      ...nextMessages,
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);
    setInput("");

    try {
      const history = nextMessages.map((m) => ({ role: m.role, content: m.content }));
      await sendChatStream(apiBase, text, history, {
        onDelta: (chunk) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + chunk } : m))
          );
        },
        onDone: (data) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content: data.answer?.direct_answer ?? m.content,
                    streaming: false,
                    responseDetail: { answer: data.answer, debug: data.debug },
                  }
                : m
            )
          );
        },
      });
    } catch (err) {
      setError(err?.message || String(err));
      setMessages((prev) => prev.filter((m) => m.id !== assistantId));
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
        <div className="headerBrand">
          <div className="title">VAWA Insights</div>
          <div className="subtitle">A Policy-Aligned Data Analysis Tool</div>
        </div>
        <div className="headerActions">
          <p id="quick-exit-hint" className="sr-only">
            Press Escape to leave this site quickly for privacy. The same action is available as the
            Quick exit button.
          </p>
          <button
            type="button"
            className="quickExit"
            onClick={quickExit}
            aria-describedby="quick-exit-hint"
            aria-label="Quick exit: leave this site for privacy"
            title="Leaves this site immediately (Escape does the same after any open panels are closed)"
          >
            Quick exit
          </button>
        </div>
      </header>

      <aside
        className="safetyDisclaimer"
        role="region"
        aria-label="Disclaimer and safety information"
      >
        <h2 className="safetyDisclaimerTitle">Disclaimer and safety</h2>
        <p>
          This chatbot is for <strong>informational and educational purposes only</strong>. It is{" "}
          <strong>not</strong> a therapist, counselor, lawyer, or emergency service, and it does not
          provide legal or clinical advice.
        </p>
        <p>
          If you are in <strong>immediate danger</strong>, call <strong>911</strong> (or your local
          emergency number).
        </p>
        <p>
          <strong>Quick exit:</strong> use the Quick exit button in the header or press{" "}
          <kbd className="kbd">Escape</kbd> (when no detail panel is open) to go to the Google
          homepage. If you are worried about someone seeing your activity, also clear your browser
          search and browsing history when it is safe to do so.
        </p>
      </aside>

      <main className="mainSingle">
        <div className="card cardChat">
          <div className="cardTitle">Chat</div>
          <div className="chatWindow" aria-label="Chat messages">
            {messages.map((m) => (
              <div key={m.id} className={`msg ${m.role === "user" ? "msgUser" : "msgAssistant"}`}>
                <div className="msgRole">{m.role}</div>
                <MessageBody role={m.role} content={m.content} streaming={Boolean(m.streaming)} />
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
import { useState, useEffect, useRef } from "react";

const TOKENS = ["The", "cat", "sat", "on", "the", "mat"];

// Simulated attention weights for each token (which other tokens it attends to)
const ATTENTION_HEADS = {
  "Syntactic (Head 1)": [
    [0.60, 0.05, 0.05, 0.10, 0.10, 0.10],
    [0.15, 0.50, 0.10, 0.05, 0.10, 0.10],
    [0.05, 0.35, 0.40, 0.05, 0.05, 0.10],
    [0.05, 0.05, 0.30, 0.40, 0.10, 0.10],
    [0.30, 0.05, 0.05, 0.05, 0.45, 0.10],
    [0.05, 0.05, 0.05, 0.25, 0.10, 0.50],
  ],
  "Semantic (Head 2)": [
    [0.35, 0.15, 0.10, 0.10, 0.15, 0.15],
    [0.10, 0.25, 0.20, 0.05, 0.05, 0.35],
    [0.05, 0.40, 0.20, 0.15, 0.05, 0.15],
    [0.10, 0.05, 0.15, 0.30, 0.10, 0.30],
    [0.25, 0.05, 0.05, 0.05, 0.30, 0.30],
    [0.05, 0.45, 0.10, 0.05, 0.05, 0.30],
  ],
  "Positional (Head 3)": [
    [0.50, 0.30, 0.10, 0.05, 0.03, 0.02],
    [0.25, 0.40, 0.20, 0.08, 0.04, 0.03],
    [0.08, 0.22, 0.35, 0.20, 0.08, 0.07],
    [0.04, 0.08, 0.20, 0.38, 0.18, 0.12],
    [0.03, 0.04, 0.08, 0.20, 0.40, 0.25],
    [0.02, 0.03, 0.05, 0.12, 0.28, 0.50],
  ],
};

const STEPS = [
  {
    id: "input",
    label: "Input & Tokenize",
    short: "Tokenize",
    description:
      'The sentence is split into tokens — roughly word-pieces. Each token becomes a discrete unit the model can process. In practice, words like "sitting" might split into "sit" + "ting".',
  },
  {
    id: "embed",
    label: "Embed + Position",
    short: "Embed",
    description:
      "Each token is mapped to a high-dimensional vector (its embedding) that encodes meaning. A positional encoding is added so the model knows word order, since attention has no built-in sense of sequence.",
  },
  {
    id: "attention",
    label: "Self-Attention",
    short: "Attention",
    description:
      'Each token computes Query, Key, and Value vectors. Attention scores = softmax(Q·Kᵀ / √d). This lets each token "look at" every other token and decide how much to blend their information.',
  },
  {
    id: "ffn",
    label: "Feed-Forward Network",
    short: "FFN",
    description:
      'After attention mixes information across positions, a small 2-layer neural network processes each token independently. This is where the model "thinks" — applying learned transformations to the attended context.',
  },
  {
    id: "output",
    label: "Output & Predict",
    short: "Output",
    description:
      "After stacking many layers of attention + FFN, the final representation is projected to vocabulary-sized logits. A softmax produces probabilities for the next token.",
  },
];

const COLORS = {
  bg: "#0f172a",
  surface: "#1e293b",
  surfaceHover: "#334155",
  border: "#334155",
  borderActive: "#6366f1",
  text: "#f1f5f9",
  textMuted: "#94a3b8",
  accent: "#6366f1",
  accentLight: "#818cf8",
  accentGlow: "rgba(99,102,241,0.15)",
  green: "#34d399",
  amber: "#fbbf24",
  rose: "#fb7185",
  cyan: "#22d3ee",
};

function heatColor(value) {
  // value 0..1 → dark blue to bright indigo/white
  const r = Math.round(30 + value * 200);
  const g = Math.round(30 + value * 80);
  const b = Math.round(80 + value * 175);
  return `rgb(${r},${g},${b})`;
}

// ─── Tiny sub-components ───

function TokenChip({ token, index, isSelected, onClick, glow, style }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "8px 16px",
        borderRadius: 8,
        border: `2px solid ${isSelected ? COLORS.accentLight : COLORS.border}`,
        background: isSelected ? COLORS.accentGlow : COLORS.surface,
        color: COLORS.text,
        fontFamily: "monospace",
        fontSize: 15,
        fontWeight: 600,
        cursor: "pointer",
        transition: "all 0.2s",
        boxShadow: glow ? `0 0 12px ${COLORS.accentLight}` : "none",
        position: "relative",
        ...style,
      }}
    >
      <span style={{ fontSize: 10, color: COLORS.textMuted, position: "absolute", top: 2, right: 6 }}>{index}</span>
      {token}
    </button>
  );
}

// ─── Step panels ───

function InputPanel() {
  return (
    <div style={{ textAlign: "center" }}>
      <p style={{ color: COLORS.textMuted, marginBottom: 16, fontSize: 14 }}>
        A sentence enters the model and is split into tokens:
      </p>
      <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
        {TOKENS.map((t, i) => (
          <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
            <TokenChip token={t} index={i} />
          </div>
        ))}
      </div>
      <div
        style={{
          marginTop: 20,
          padding: 14,
          background: COLORS.surface,
          borderRadius: 8,
          border: `1px solid ${COLORS.border}`,
          fontSize: 13,
          color: COLORS.textMuted,
        }}
      >
        <strong style={{ color: COLORS.amber }}>Try it:</strong> In real models, tokenization uses algorithms like BPE (Byte-Pair Encoding). "unhappiness" might become ["un", "happi", "ness"].
      </div>
    </div>
  );
}

function EmbedPanel() {
  const dims = [
    [0.21, -0.85, 0.43, 0.67],
    [-0.33, 0.72, -0.15, 0.91],
    [0.55, 0.12, -0.78, 0.34],
    [-0.44, -0.22, 0.88, -0.11],
    [0.19, -0.82, 0.41, 0.65],
    [0.61, 0.45, -0.67, 0.23],
  ];
  const pos = [
    [0.00, 1.00, 0.00, 1.00],
    [0.84, 0.54, 0.01, 1.00],
    [0.91, -0.42, 0.02, 1.00],
    [0.14, -0.99, 0.03, 0.99],
    [-0.76, -0.65, 0.04, 0.99],
    [-0.96, 0.28, 0.05, 0.99],
  ];

  return (
    <div>
      <p style={{ color: COLORS.textMuted, marginBottom: 12, fontSize: 14, textAlign: "center" }}>
        Each token maps to a <strong style={{ color: COLORS.cyan }}>semantic embedding</strong> + a <strong style={{ color: COLORS.green }}>positional encoding</strong> (showing 4 of ~768 dimensions):
      </p>
      <div style={{ overflowX: "auto" }}>
        <table style={{ margin: "0 auto", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr>
              <th style={{ padding: "6px 12px", color: COLORS.textMuted }}>Token</th>
              <th style={{ padding: "6px 12px", color: COLORS.cyan }}>Embedding</th>
              <th style={{ padding: "6px 8px", color: COLORS.textMuted }}>+</th>
              <th style={{ padding: "6px 12px", color: COLORS.green }}>Position</th>
              <th style={{ padding: "6px 8px", color: COLORS.textMuted }}>=</th>
              <th style={{ padding: "6px 12px", color: COLORS.accentLight }}>Combined</th>
            </tr>
          </thead>
          <tbody>
            {TOKENS.map((t, i) => (
              <tr key={i}>
                <td style={{ padding: "4px 12px", fontFamily: "monospace", color: COLORS.text, fontWeight: 600 }}>{t}</td>
                <td style={{ padding: "4px 12px", fontFamily: "monospace", color: COLORS.cyan, fontSize: 12 }}>
                  [{dims[i].map((d) => d.toFixed(2)).join(", ")}]
                </td>
                <td style={{ color: COLORS.textMuted, textAlign: "center" }}>+</td>
                <td style={{ padding: "4px 12px", fontFamily: "monospace", color: COLORS.green, fontSize: 12 }}>
                  [{pos[i].map((d) => d.toFixed(2)).join(", ")}]
                </td>
                <td style={{ color: COLORS.textMuted, textAlign: "center" }}>=</td>
                <td style={{ padding: "4px 12px", fontFamily: "monospace", color: COLORS.accentLight, fontSize: 12 }}>
                  [{dims[i].map((d, j) => (d + pos[i][j]).toFixed(2)).join(", ")}]
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AttentionPanel() {
  const [selectedToken, setSelectedToken] = useState(1); // "cat"
  const [headName, setHeadName] = useState("Syntactic (Head 1)");
  const weights = ATTENTION_HEADS[headName][selectedToken];

  return (
    <div>
      <p style={{ color: COLORS.textMuted, marginBottom: 12, fontSize: 14, textAlign: "center" }}>
        Click a token to see what it <strong style={{ color: COLORS.accentLight }}>attends to</strong>. Switch heads to see different relationship types.
      </p>

      {/* Head selector */}
      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 16, flexWrap: "wrap" }}>
        {Object.keys(ATTENTION_HEADS).map((h) => (
          <button
            key={h}
            onClick={() => setHeadName(h)}
            style={{
              padding: "6px 14px",
              borderRadius: 20,
              border: `1px solid ${h === headName ? COLORS.accentLight : COLORS.border}`,
              background: h === headName ? COLORS.accentGlow : "transparent",
              color: h === headName ? COLORS.accentLight : COLORS.textMuted,
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {h}
          </button>
        ))}
      </div>

      {/* Source token selector */}
      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 20, flexWrap: "wrap" }}>
        {TOKENS.map((t, i) => (
          <TokenChip key={i} token={t} index={i} isSelected={i === selectedToken} onClick={() => setSelectedToken(i)} glow={i === selectedToken} />
        ))}
      </div>

      {/* Attention visualization */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6, flexWrap: "wrap" }}>
        {TOKENS.map((t, i) => {
          const w = weights[i];
          return (
            <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <div
                style={{
                  width: 60,
                  height: 50,
                  borderRadius: 8,
                  background: heatColor(w),
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontFamily: "monospace",
                  fontSize: 13,
                  fontWeight: 700,
                  color: w > 0.3 ? "#fff" : COLORS.textMuted,
                  border: `2px solid ${w > 0.3 ? COLORS.accentLight : COLORS.border}`,
                  transition: "all 0.3s",
                }}
              >
                {t}
              </div>
              <div
                style={{
                  height: 6,
                  width: 50,
                  borderRadius: 3,
                  background: COLORS.surface,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${w * 100}%`,
                    background: `linear-gradient(90deg, ${COLORS.accent}, ${COLORS.accentLight})`,
                    borderRadius: 3,
                    transition: "width 0.3s",
                  }}
                />
              </div>
              <span style={{ fontSize: 11, color: COLORS.textMuted, fontFamily: "monospace" }}>{(w * 100).toFixed(0)}%</span>
            </div>
          );
        })}
      </div>
      <p style={{ textAlign: "center", fontSize: 12, color: COLORS.textMuted, marginTop: 12 }}>
        <strong style={{ color: COLORS.text }}>"{TOKENS[selectedToken]}"</strong> attending via <strong style={{ color: COLORS.accentLight }}>{headName}</strong> — brighter = stronger attention
      </p>
    </div>
  );
}

function FFNPanel() {
  const [activeNeuron, setActiveNeuron] = useState(null);
  const neurons = [
    { label: "subject?", activation: 0.92, note: "Detects subject-verb pattern" },
    { label: "location?", activation: 0.15, note: "Low — no location signal here" },
    { label: "animal?", activation: 0.87, note: "Strong — 'cat' activated this" },
    { label: "action?", activation: 0.78, note: "'sat' triggered action detector" },
    { label: "negation?", activation: 0.04, note: "No negation present" },
    { label: "plural?", activation: 0.11, note: "Singular noun detected" },
  ];

  return (
    <div>
      <p style={{ color: COLORS.textMuted, marginBottom: 16, fontSize: 14, textAlign: "center" }}>
        Each token passes through a 2-layer network independently. Hover over neurons to see what features they detect:
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, maxWidth: 420, margin: "0 auto" }}>
        {neurons.map((n, i) => (
          <button
            key={i}
            onMouseEnter={() => setActiveNeuron(i)}
            onMouseLeave={() => setActiveNeuron(null)}
            style={{
              padding: "12px 8px",
              borderRadius: 10,
              border: `1px solid ${activeNeuron === i ? COLORS.accentLight : COLORS.border}`,
              background: COLORS.surface,
              cursor: "pointer",
              textAlign: "center",
              transition: "all 0.2s",
              transform: activeNeuron === i ? "scale(1.05)" : "scale(1)",
            }}
          >
            <div style={{ fontSize: 12, color: COLORS.textMuted, marginBottom: 4 }}>{n.label}</div>
            <div
              style={{
                height: 8,
                borderRadius: 4,
                background: COLORS.bg,
                overflow: "hidden",
                marginBottom: 4,
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${n.activation * 100}%`,
                  background:
                    n.activation > 0.5
                      ? `linear-gradient(90deg, ${COLORS.green}, ${COLORS.cyan})`
                      : `linear-gradient(90deg, ${COLORS.border}, ${COLORS.textMuted})`,
                  borderRadius: 4,
                }}
              />
            </div>
            <div style={{ fontSize: 13, fontWeight: 700, color: n.activation > 0.5 ? COLORS.green : COLORS.textMuted, fontFamily: "monospace" }}>
              {(n.activation * 100).toFixed(0)}%
            </div>
          </button>
        ))}
      </div>
      {activeNeuron !== null && (
        <p style={{ textAlign: "center", marginTop: 12, fontSize: 13, color: COLORS.amber, transition: "all 0.2s" }}>
          {neurons[activeNeuron].note}
        </p>
      )}
      <p style={{ textAlign: "center", marginTop: 12, fontSize: 12, color: COLORS.textMuted }}>
        Real FFNs have thousands of neurons. ReLU activation zeros out irrelevant features and amplifies important ones.
      </p>
    </div>
  );
}

function OutputPanel() {
  const predictions = [
    { token: ".", prob: 0.42 },
    { token: "today", prob: 0.18 },
    { token: "quietly", prob: 0.14 },
    { token: "and", prob: 0.09 },
    { token: "comfortably", prob: 0.06 },
  ];
  const [sampled, setSampled] = useState(null);
  const [temp, setTemp] = useState(1.0);

  function sampleToken() {
    // Simple temperature-adjusted sampling
    const logits = predictions.map((p) => Math.log(p.prob));
    const adjusted = logits.map((l) => l / temp);
    const maxL = Math.max(...adjusted);
    const exps = adjusted.map((a) => Math.exp(a - maxL));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((e) => e / sum);
    const r = Math.random();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (r < cum) {
        setSampled(predictions[i].token);
        return;
      }
    }
    setSampled(predictions[predictions.length - 1].token);
  }

  return (
    <div style={{ textAlign: "center" }}>
      <p style={{ color: COLORS.textMuted, marginBottom: 16, fontSize: 14 }}>
        The model predicts probability for each possible next token. Adjust <strong style={{ color: COLORS.amber }}>temperature</strong> and sample:
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: 8, maxWidth: 340, margin: "0 auto 16px" }}>
        {predictions.map((p, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontFamily: "monospace", fontSize: 14, color: COLORS.text, width: 90, textAlign: "right", fontWeight: sampled === p.token ? 700 : 400 }}>
              "{p.token}"
            </span>
            <div style={{ flex: 1, height: 20, background: COLORS.surface, borderRadius: 6, overflow: "hidden" }}>
              <div
                style={{
                  height: "100%",
                  width: `${p.prob * 100 * 2}%`,
                  background: sampled === p.token ? `linear-gradient(90deg, ${COLORS.green}, ${COLORS.cyan})` : `linear-gradient(90deg, ${COLORS.accent}, ${COLORS.accentLight})`,
                  borderRadius: 6,
                  transition: "all 0.3s",
                }}
              />
            </div>
            <span style={{ fontFamily: "monospace", fontSize: 12, color: COLORS.textMuted, width: 40 }}>{(p.prob * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 12, marginBottom: 16 }}>
        <span style={{ fontSize: 12, color: COLORS.textMuted }}>Temperature:</span>
        <input
          type="range"
          min="0.1"
          max="2.0"
          step="0.1"
          value={temp}
          onChange={(e) => setTemp(parseFloat(e.target.value))}
          style={{ width: 120, accentColor: COLORS.accent }}
        />
        <span style={{ fontFamily: "monospace", fontSize: 13, color: COLORS.amber, width: 30 }}>{temp.toFixed(1)}</span>
      </div>

      <button
        onClick={sampleToken}
        style={{
          padding: "10px 28px",
          borderRadius: 8,
          border: "none",
          background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.accentLight})`,
          color: "#fff",
          fontWeight: 700,
          fontSize: 14,
          cursor: "pointer",
          boxShadow: `0 4px 15px ${COLORS.accentGlow}`,
        }}
      >
        Sample Next Token
      </button>
      {sampled && (
        <p style={{ marginTop: 12, fontSize: 15, color: COLORS.green, fontWeight: 600 }}>
          "The cat sat on the mat" → <strong style={{ color: COLORS.cyan }}>"{sampled}"</strong>
        </p>
      )}
      <p style={{ fontSize: 11, color: COLORS.textMuted, marginTop: 8 }}>
        Low temperature → picks the most likely token. High temperature → more random/creative.
      </p>
    </div>
  );
}

// ─── Architecture Diagram ───

function ArchDiagram({ currentStep }) {
  const layers = [
    { id: "input", label: "Tokens", color: COLORS.cyan },
    { id: "embed", label: "Embed + Pos", color: COLORS.green },
    { id: "attention", label: "Self-Attention", color: COLORS.accentLight },
    { id: "ffn", label: "Feed-Forward", color: COLORS.amber },
    { id: "output", label: "Output", color: COLORS.rose },
  ];

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 4, margin: "16px 0", flexWrap: "wrap" }}>
      {layers.map((l, i) => (
        <div key={l.id} style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <div
            style={{
              padding: "6px 14px",
              borderRadius: 6,
              background: currentStep === l.id ? l.color + "22" : COLORS.surface,
              border: `2px solid ${currentStep === l.id ? l.color : COLORS.border}`,
              color: currentStep === l.id ? l.color : COLORS.textMuted,
              fontSize: 12,
              fontWeight: 700,
              transition: "all 0.3s",
              boxShadow: currentStep === l.id ? `0 0 10px ${l.color}44` : "none",
            }}
          >
            {l.label}
          </div>
          {i < layers.length - 1 && <span style={{ color: COLORS.textMuted, fontSize: 16 }}>→</span>}
        </div>
      ))}
    </div>
  );
}

// ─── Main App ───

export default function TransformerDemo() {
  const [stepIdx, setStepIdx] = useState(0);
  const step = STEPS[stepIdx];

  const panels = {
    input: <InputPanel />,
    embed: <EmbedPanel />,
    attention: <AttentionPanel />,
    ffn: <FFNPanel />,
    output: <OutputPanel />,
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: COLORS.bg,
        color: COLORS.text,
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        padding: "24px 16px",
      }}
    >
      <div style={{ maxWidth: 680, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 20 }}>
          <h1 style={{ fontSize: 26, fontWeight: 800, margin: 0, letterSpacing: "-0.5px" }}>
            <span style={{ color: COLORS.accentLight }}>Transformer</span> Architecture
          </h1>
          <p style={{ color: COLORS.textMuted, fontSize: 14, margin: "6px 0 0" }}>
            Interactive walkthrough — click each stage to explore
          </p>
        </div>

        {/* Architecture flow */}
        <ArchDiagram currentStep={step.id} />

        {/* Step tabs */}
        <div style={{ display: "flex", gap: 4, justifyContent: "center", margin: "16px 0", flexWrap: "wrap" }}>
          {STEPS.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setStepIdx(i)}
              style={{
                padding: "8px 16px",
                borderRadius: 8,
                border: `1px solid ${i === stepIdx ? COLORS.accentLight : COLORS.border}`,
                background: i === stepIdx ? COLORS.accentGlow : "transparent",
                color: i === stepIdx ? COLORS.accentLight : COLORS.textMuted,
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.2s",
              }}
            >
              {i + 1}. {s.short}
            </button>
          ))}
        </div>

        {/* Step description */}
        <div
          style={{
            padding: "14px 18px",
            background: COLORS.surface,
            borderRadius: 10,
            border: `1px solid ${COLORS.border}`,
            marginBottom: 20,
          }}
        >
          <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 6px", color: COLORS.accentLight }}>
            Step {stepIdx + 1}: {step.label}
          </h2>
          <p style={{ fontSize: 13, color: COLORS.textMuted, margin: 0, lineHeight: 1.6 }}>{step.description}</p>
        </div>

        {/* Interactive panel */}
        <div
          style={{
            padding: 20,
            background: COLORS.surface,
            borderRadius: 12,
            border: `1px solid ${COLORS.border}`,
            minHeight: 200,
          }}
        >
          {panels[step.id]}
        </div>

        {/* Navigation */}
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16 }}>
          <button
            onClick={() => setStepIdx(Math.max(0, stepIdx - 1))}
            disabled={stepIdx === 0}
            style={{
              padding: "8px 20px",
              borderRadius: 8,
              border: `1px solid ${COLORS.border}`,
              background: "transparent",
              color: stepIdx === 0 ? COLORS.border : COLORS.textMuted,
              fontSize: 13,
              cursor: stepIdx === 0 ? "default" : "pointer",
            }}
          >
            ← Previous
          </button>
          <button
            onClick={() => setStepIdx(Math.min(STEPS.length - 1, stepIdx + 1))}
            disabled={stepIdx === STEPS.length - 1}
            style={{
              padding: "8px 20px",
              borderRadius: 8,
              border: `1px solid ${stepIdx === STEPS.length - 1 ? COLORS.border : COLORS.accentLight}`,
              background: stepIdx === STEPS.length - 1 ? "transparent" : COLORS.accentGlow,
              color: stepIdx === STEPS.length - 1 ? COLORS.border : COLORS.accentLight,
              fontSize: 13,
              fontWeight: 600,
              cursor: stepIdx === STEPS.length - 1 ? "default" : "pointer",
            }}
          >
            Next →
          </button>
        </div>

        {/* Formula card */}
        <div
          style={{
            marginTop: 20,
            padding: "14px 18px",
            background: COLORS.bg,
            borderRadius: 10,
            border: `1px dashed ${COLORS.border}`,
            textAlign: "center",
          }}
        >
          <p style={{ fontSize: 12, color: COLORS.textMuted, margin: "0 0 6px" }}>The core formula:</p>
          <code style={{ fontSize: 15, color: COLORS.amber, fontWeight: 700 }}>
            Attention(Q, K, V) = softmax(Q·Kᵀ / √dₖ) · V
          </code>
        </div>
      </div>
    </div>
  );
}

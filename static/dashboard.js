/**
 * StoryWorld UI — Infrastructure console.
 * INVARIANT: The UI must NEVER require video playback to understand outcomes.
 * Outcome, confidence, WorldStateGraph, and constraints are sufficient.
 */
const API_BASE = "";

function escapeHtml(s) {
    if (s == null) return "";
    const div = document.createElement("div");
    div.textContent = String(s);
    return div.innerHTML;
}

function formatTimeAgo(iso) {
    if (!iso) return "-";
    const d = new Date(iso);
    const now = new Date();
    const sec = Math.floor((now - d) / 1000);
    if (sec < 60) return "just now";
    if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
    if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
    return `${Math.floor(sec / 86400)}d ago`;
}

async function loadDashboard() {
    const tbody = document.getElementById("sim-table-body");
    if (!tbody) return;
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 10000);
        const res = await fetch("/episodes?limit=20", { signal: controller.signal });
        clearTimeout(timeout);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        const episodes = json.episodes || [];
        const data = episodes.map(ep => ({
            id: ep.episode_id,
            goal: ep.intent || "(no goal)",
            status: (ep.state || "CREATED").toLowerCase(),
            cost: "-",
            confidence: "-",
            time: formatTimeAgo(ep.updated_at),
        }));

        if (data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding: 40px; color: var(--text-muted);">No simulations yet. <a href="/new.html">Create one</a>.</td></tr>';
            return;
        }
        tbody.innerHTML = data.map(sim => `
            <tr>
                <td class="mono"><a href="/simulation.html?id=${sim.id}" class="text-link">${sim.id}</a></td>
                <td style="font-weight:500; color:var(--text-primary);">${sim.goal}</td>
                <td><span class="status status-${sim.status}">${sim.status.toUpperCase()}</span></td>
                <td class="mono">${sim.confidence}</td>
                <td class="mono">${sim.cost}</td>
                <td class="mono" style="color:var(--text-muted);">${sim.time}</td>
                <td style="text-align:right;">
                    <a href="/simulation.html?id=${sim.id}" style="color:var(--text-secondary); text-decoration:none;">&rarr;</a>
                </td>
            </tr>
        `).join("");
    } catch (e) {
        console.error("Dashboard error", e);
        if (tbody) tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding: 40px; color: var(--text-muted);">Failed to load. <a href="/episodes?limit=20" target="_blank">Test API</a> or check console.</td></tr>';
    }
}

async function loadDetail() {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id") || ""; 
    if (!id) {
        document.getElementById("sim-id").textContent = "No ID";
        return;
    }
    
    document.getElementById("sim-id").textContent = id;

    // Polling function
    const fetchStatus = async () => {
        try {
            const res = await fetch(`/episodes/${id}`);
            if (!res.ok) throw new Error("Not found");
            const data = await res.json();
            
            // Fetch world state for state nodes / transitions
            try {
                const wsRes = await fetch(`/world-state/${id}`);
                if (wsRes.ok) {
                    const ws = await wsRes.json();
                    data.state_nodes = ws.total_nodes || 0;
                    data.transitions = ws.total_transitions || 0;
                    data.ws_nodes = ws.nodes || [];
                    data.ws_transitions = ws.transitions || [];
                }
            } catch (_) {}
            
            // When terminal, fetch result for outcome, constraints_discovered
            const status = (data.state || data.status || "").toLowerCase();
            const isTerminal = ["completed", "failed", "goal_impossible", "partially_completed", "dead_state", "goal_abandoned", "epistemically_blocked", "epistemically_incomplete"].includes(status);
            if (isTerminal) {
                try {
                    const resultRes = await fetch(`/episodes/${id}/result?include_video=false`);
                    if (resultRes.ok) {
                        const result = await resultRes.json();
                        data.outcome = result.outcome;
                        data.constraints_discovered = result.constraints_discovered || [];
                        data.confidence = result.confidence ?? data.confidence;
                        data.total_cost_usd = result.total_cost_usd ?? data.budget_spent_usd;
                        data.verdict_explanation = result.state_delta?.verdict_explanation || null;
                        data.suggested_alternatives = result.suggested_alternatives || [];
                        data.attempts_made = result.attempts_made || [];
                    }
                } catch (_) {}
            }
            
            updateDetailUI(data);

            return isTerminal;
        } catch (e) {
            document.getElementById("json-state").textContent = "Connecting to simulation stream...";
            return false;
        }
    };

    // Initial load
    const finished = await fetchStatus();
    if (!finished) {
        // Poll every 2 seconds
        const interval = setInterval(async () => {
            const isDone = await fetchStatus();
            if (isDone) clearInterval(interval);
        }, 2000);
    }
}

function updateDetailUI(data) {
    // Prefer outcome from /result when terminal (e.g. goal_impossible vs partially_completed)
    const rawState = data.state || data.status || "PENDING";
    const rawOutcome = (data.outcome || "").toLowerCase().replace(/-/g, "_");
    const status = (rawOutcome && ["goal_achieved", "goal_impossible", "goal_abandoned", "dead_state"].includes(rawOutcome))
        ? rawOutcome
        : rawState.toLowerCase();
    const badge = document.getElementById("outcome-badge");
    badge.textContent = (data.outcome || rawState).toUpperCase().replace(/_/g, " ");
    badge.className = `status status-${status}`;
    
    document.getElementById("confidence").textContent = data.confidence != null ? String(data.confidence) : "-";
    const cost = data.total_cost_usd ?? data.budget_spent_usd ?? 0;
    document.getElementById("total-cost").textContent = `$${Number(cost).toFixed(2)}`;
    
    const sn = document.getElementById("state-nodes");
    const tr = document.getElementById("transitions");
    if (sn) sn.textContent = data.state_nodes != null ? data.state_nodes : "-";
    if (tr) tr.textContent = data.transitions != null ? data.transitions : "-";
    
    // Verdict explanation panel (explainability)
    const explPanel = document.getElementById("verdict-explanation");
    const explContent = document.getElementById("verdict-explanation-content");
    if (explPanel && explContent) {
        if (data.verdict_explanation) {
            const v = data.verdict_explanation;
            explPanel.style.display = "block";
            explContent.innerHTML = `
                <div class="explanation-summary" style="font-weight:500; margin-bottom:8px;">${escapeHtml(v.summary || `Verdict: ${v.verdict}`)}</div>
                ${(v.causal_chain || []).length ? `<ul style="margin:0; padding-left:18px; color:var(--text-secondary);">${(v.causal_chain || []).map(c => `<li>${escapeHtml(c)}</li>`).join("")}</ul>` : ""}
                ${(v.missing_evidence || []).length ? `<div style="margin-top:8px; font-size:12px; color:var(--accent, #f59e0b);">Missing evidence: ${escapeHtml((v.missing_evidence || []).join(", "))}</div>` : ""}
                ${(v.constraints_inferred || []).length ? `<div style="margin-top:8px; font-size:12px; color:var(--text-muted);">Constraints inferred: ${escapeHtml((v.constraints_inferred || []).join(", "))}</div>` : ""}
            `;
        } else {
            explPanel.style.display = "none";
        }
    }

    // Exploratory suggestions panel
    const expPanel = document.getElementById("exploratory-suggestions");
    const altContent = document.getElementById("suggested-alternatives-content");
    const attemptsContent = document.getElementById("attempts-made-content");
    if (expPanel && altContent && attemptsContent) {
        const alts = data.suggested_alternatives || [];
        const attempts = data.attempts_made || [];
        if (alts.length > 0 || attempts.length > 0) {
            expPanel.style.display = "block";
            altContent.innerHTML = alts.length > 0
                ? `<ul style="margin:0; padding-left:18px; color:var(--text-secondary);">${alts.map(a => `<li>${escapeHtml(a)}</li>`).join("")}</ul>`
                : "";
            attemptsContent.innerHTML = attempts.length > 0
                ? `<div style="font-size:12px; color:var(--text-muted);">Framings tried: ${attempts.map(a => `#${a.observability_attempt} ${escapeHtml(a.render_hint || "(default)")}`).join(" · ")}</div>`
                : "";
        } else {
            expPanel.style.display = "none";
        }
    }

    // State-first display: outcome, constraints, progress. Beats minimized (no cinematic bloat).
    const displayData = {
        state: data.state,
        outcome: data.outcome,
        confidence: data.confidence,
        cost: data.total_cost_usd ?? data.budget_spent_usd,
        constraints_discovered: data.constraints_discovered,
        verdict_explanation: data.verdict_explanation,
        suggested_alternatives: data.suggested_alternatives,
        attempts_made: data.attempts_made,
        progress: data.progress,
        state_nodes: data.state_nodes,
        transitions: data.transitions,
    };
    if ((!data.beats || data.beats.length === 0) && (data.state || "").toLowerCase() === "impossible") {
        displayData.note = "Early veto: deterministic physics constraint (no render needed)";
    }
    if (data.beats) {
        displayData.beats = data.beats.map(b => ({
            beat_id: b.beat_id,
            description: (b.description || b.spec?.description || "").substring(0, 80),
            state: b.state,
        }));
    }
    document.getElementById("json-state").textContent = JSON.stringify(displayData, null, 2);

    // Update Graph
    renderGraph(data);
}

function renderGraph(data) {
    const graphDiv = document.getElementById("world-graph");
    if (!graphDiv) return;

    const sanitize = (value) => {
        const text = String(value ?? "").replace(/["\n\r]/g, " ").trim();
        return text.length > 60 ? text.slice(0, 60) + "…" : text;
    };

    let mermaidCode = "graph LR\n";
    mermaidCode += "Start((Start)) --> ";

    const transitions = data.ws_transitions || [];
    const nodes = data.ws_nodes || [];
    if (transitions.length > 0 && nodes.length > 0) {
        // Render actual WorldStateGraph transitions
        const nodeLabel = (n) => {
            const depth = n.depth != null ? `S${n.depth}` : "State";
            return sanitize(depth);
        };
        const idMap = {};
        nodes.forEach((n, idx) => {
            const nid = `N${idx}`;
            idMap[n.node_id] = nid;
            mermaidCode += `${nid}["${nodeLabel(n)}"]\n`;
        });
        transitions.forEach((t, idx) => {
            const src = idMap[t.source_node_id] || "Start";
            const dst = idMap[t.target_node_id] || `T${idx}`;
            if (!idMap[t.target_node_id]) {
                mermaidCode += `${dst}["Rejected"]\n`;
            }
            let verdict = "";
            try {
                if (t.observation_json) {
                    const obs = JSON.parse(t.observation_json);
                    verdict = obs.verdict ? ` ${sanitize(obs.verdict)}` : "";
                }
            } catch (_) {}
            const status = sanitize((t.status || "").toUpperCase());
            mermaidCode += `${src} -->|${status}${verdict}| ${dst}\n`;
        });
    } else {
        const beats = data.beats || [];
        if (beats.length === 0) {
            mermaidCode += "Init[Initializing...]";
        } else {
        // Simple linear chain for now, representing episode beats
        let prevNode = "Start";
        beats.forEach((beat, index) => {
            const nodeId = `Beat${index}`;
            // NOTE: sql_store unpacks the 'spec' JSON, so 'description' is top-level.
            const desc = beat.description || beat.spec?.description || `Beat ${index+1}`;
            const label = desc.substring(0, 20) + "...";
            const state = beat.state || "PENDING";
            
            // Color based on state
            let style = "";
            if (state === "ACCEPTED") style = "style " + nodeId + " fill:#22c55e,stroke:#22c55e,color:#000";
            else if (state === "ABORTED") style = "style " + nodeId + " fill:#ef4444,stroke:#ef4444,color:#fff";
            else style = "style " + nodeId + " fill:#333,stroke:#666,color:#fff";

            mermaidCode += `${nodeId}["${label}"]\n`;
            if (prevNode !== nodeId) mermaidCode += `${prevNode} --> ${nodeId}\n`;
            mermaidCode += style + "\n";
            
            prevNode = nodeId;
        });

            var terminalState = (data.state || data.status || "").toLowerCase();
            if (terminalState === "completed") {
                mermaidCode += `${prevNode} --> End((GOAL_ACHIEVED))\nstyle End fill:#22c55e,stroke:#fff,color:#000`;
            } else if (terminalState === "goal_impossible" || terminalState === "dead_state") {
                mermaidCode += `${prevNode} --> Imp((GOAL_IMPOSSIBLE))\nstyle Imp fill:#f59e0b,stroke:#fff,color:#000`;
            } else if (terminalState === "failed" || terminalState === "partially_completed" || terminalState === "goal_abandoned") {
                mermaidCode += `${prevNode} --> Fail((GOAL_ABANDONED))\nstyle Fail fill:#ef4444,stroke:#fff,color:#fff`;
            }
        }
    }

    // Insert and render
    graphDiv.innerHTML = mermaidCode;
    graphDiv.removeAttribute('data-processed'); // Reset for re-render
    try {
        mermaid.init(undefined, graphDiv);
    } catch(e) { console.warn("Mermaid render error", e); }
}

// Handlers for New Simulation Form
async function submitSimulation(e) {
    e.preventDefault();
    const form = e.target;
    const goal = (form.querySelector('#goal') || form.querySelector('textarea')).value.trim();
    const budget = parseFloat((form.querySelector('#budget') || form.querySelector('input[type="number"]'))?.value || 5);
    const riskProfile = (form.querySelector('#risk_profile') || form.querySelector('select'))?.value || 'medium';
    
    const btn = form.querySelector('button[type="submit"]');
    const originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Provisioning...";

    try {
        const params = new URLSearchParams({ world_id: 'default', goal });
        if (!isNaN(budget) && budget > 0) params.set('budget', String(budget));
        if (riskProfile) params.set('risk_profile', riskProfile);
        const res = await fetch(`/simulate?${params}`, { method: "POST" });
        
        if (!res.ok) throw new Error("API Error: " + res.status);
        
        const data = await res.json();
        window.location.href = `/simulation.html?id=${data.simulation_id}`;
    } catch (err) {
        alert("Failed to start simulation: " + err.message);
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

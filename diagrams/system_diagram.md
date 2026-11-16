# System Diagram

Derived from the code in `app.py` and `src/`. The first diagram is the short overview: the main components, how data flows from input to output, and where tests and people check the AI. The two after it are detail views of the same system.

## Overview

```mermaid
---
config:
  flowchart:
    wrappingWidth: 260
---
flowchart LR
    U([User<br/>free text or sliders]) --> A
    A["<b>1. BUILD PROFILE</b><br/>Parser: Gemini<br/>Guardrail validator"] --> B
    B["<b>2. RANK</b><br/>Scorer: no AI<br/>240-song catalog"] --> C
    C["<b>3. EXPLAIN</b><br/>Retriever: TF-IDF<br/>Explainer: Gemini"] --> D
    D["<b>4. VERIFY</b><br/>Faithfulness checker<br/>Confidence score"] --> E
    E([<b>OUTPUT</b><br/>Songs, explanations,<br/>confidence warning])

    PT["Pytest<br/>238 offline tests"] -.-> B
    EV["Eval harness<br/>+ mutation test"] -.-> C
    HUM["Human review<br/>of explanations"] -.-> E

    classDef stage fill:#ecfdf5,stroke:#2dd4bf,stroke-width:2px
    classDef ends fill:#fdf2ff,stroke:#c084fc,stroke-width:2px
    classDef check fill:#eff6ff,stroke:#60a5fa,stroke-width:2px
    class A,B,C,D stage
    class U,E ends
    class PT,EV,HUM check
```

- **Components (left to right, as data flows):** the parser (Gemini) turns text into a profile, the guardrail validates it, the scorer ranks the songs, the retriever finds notes for each result, the explainer (Gemini) writes the reasons, the faithfulness checker verifies them, and the confidence score summarizes how far to trust the result.
- **Data flow:** a request or slider settings go in, a ranked list with explanations and a confidence level comes out. Gemini never ranks songs.
- **Where the AI is checked:** the guardrail and the faithfulness checker run on every request; pytest and the eval harness run on demand; a person reviews explanations, notes and the run log.

## Detail 1: Request flow (input, process, output)

Stages 1 and 2 first, then stages 3 to 5.

**1a. Build the profile and rank**

```mermaid
flowchart LR
    U([User]) -->|free text| PAR
    U -->|sliders and dropdowns| SLD

    subgraph G1 ["1. Build the profile"]
        PAR["Parser (llm_parser.py)<br/>Gemini: genre or null<br/>+ 3 numbers<br/>mood computed from numbers"]
        VAL{{"Guardrail<br/>validate_profile<br/>+ labels exist<br/>in the catalog"}}
        SLD["Slider profile<br/>same validator,<br/>always valid"]
        PAR --> VAL
        PAR -.->|"API error, 429,<br/>no music preference"| SLD
        VAL -.->|invalid| SLD
    end

    subgraph G2 ["2. Rank (no AI)"]
        SC["Scorer (recommender.py)<br/>weighted closeness,<br/>top-k, reasons"]
        CSV[("songs.csv<br/>240 songs")]
        CSV --> SC
    end

    VAL -->|valid| SC
    SLD --> SC
    SC ==>|"ranked songs<br/>go to stage 3"| NEXT(["to 1b"])

    style G1 fill:#fdf2ff,stroke:#c084fc
    style G2 fill:#fff7ed,stroke:#fb923c
```

**1b. Explain, verify, show**

```mermaid
flowchart LR
    PREV(["from 1a:<br/>ranked songs"]) --> RET
    PREV -.->|top-k scores| CONF

    subgraph G3 ["3. Explain (RAG)"]
        RET["Retriever (retriever.py)<br/>TF-IDF notes<br/>per result song"]
        KN[("notes.json<br/>38 notes")]
        EXP["Explainer (explainer.py)<br/>one batched<br/>Gemini call"]
        KN --> RET
        RET -->|notes per song| EXP
    end

    subgraph G4 ["4. Verify"]
        CHK{{"Faithfulness checker<br/>(reliability.py)<br/>rule-based,<br/>per explanation"}}
        FB["Fallback sentence<br/>from the score<br/>breakdown"]
        CONF["Confidence score<br/>high, medium, low"]
        CHK -->|fails| FB
        CHK -->|passes| CONF
        FB --> CONF
    end

    subgraph G5 ["5. Show and log"]
        OUT["Streamlit UI (app.py)<br/>songs, explanations,<br/>breakdown, confidence,<br/>low-confidence warning"]
        LOG[("Run log<br/>logs/run.jsonl")]
        OUT -.-> LOG
    end

    EXP -->|text per song| CHK
    EXP -.->|"no API key, error,<br/>or missing text"| FB
    CONF --> OUT

    style G3 fill:#ecfdf5,stroke:#2dd4bf
    style G4 fill:#fefce8,stroke:#facc15
    style G5 fill:#eff6ff,stroke:#60a5fa
```

**Reading the diagram**

- **The five groups are the stages.** Group 1 has two ways in and one guardrail. Text goes through Gemini first; sliders skip the model. Both end in `validate_profile`.
- **Gemini never ranks songs.** It only turns text into a profile (stage 1) and writes explanation text (stage 3). Ranking is plain arithmetic.
- **Every AI step has a non-AI fallback.** A failed parse uses the sliders. A failed or missing explanation uses a sentence built from the score breakdown.
- **Order matters.** Retrieval runs after scoring and looks up each result song's own labels, not the user's words. Confidence is computed last, so it can include how many explanations passed the check.
- **Without an API key** the app still runs: sliders, scoring, fallback explanations and confidence all work.

## Detail 2: Testing and human review

```mermaid
flowchart LR
    subgraph G1 ["1. System under test"]
        CODE["Code<br/>src/ and app.py"]
        GEM["Gemini<br/>(free tier)"]
    end

    subgraph G2 ["2. Automatic checks (on demand)"]
        PT["Pytest<br/>238 offline tests<br/>Gemini mocked and blocked"]
        EV["Evaluation harness<br/>eval/run_eval.py<br/>guardrail, 90-profile ranking sweep,<br/>confidence, parser, off-topic,<br/>consistency, explanations"]
        MUT["Checker mutation test<br/>corrupt real explanations,<br/>count what the checker catches"]
        EV --> MUT
    end

    subgraph G3 ["3. Evidence"]
        REP[("eval/results/*.json<br/>saved reports")]
        LOG[("logs/run.jsonl<br/>rejected text, failures,<br/>confidence")]
    end

    subgraph G4 ["4. Human review"]
        HUM["The developer<br/>reads explanations against breakdowns,<br/>checks notes, reads rejected texts<br/>and the log, judges bias findings"]
        CHG["Changes<br/>prompts, notes, checker rules,<br/>weights, catalog build<br/>then re-run, average several live runs"]
        HUM -.->|decide what to change| CHG
    end

    CODE --> PT
    CODE --> EV
    GEM <-->|"live calls,<br/>6 s apart"| EV
    EV -->|"API errors counted<br/>separately"| REP
    MUT --> REP
    REP --> HUM
    LOG --> HUM

    style G1 fill:#fdf2ff,stroke:#c084fc
    style G2 fill:#ecfdf5,stroke:#2dd4bf
    style G3 fill:#fff7ed,stroke:#fb923c
    style G4 fill:#eff6ff,stroke:#60a5fa
```

**Reading the diagram**

- **Pytest** checks the deterministic code and the AI wrappers without any network. It cannot say how Gemini really behaves.
- **The evaluation harness** calls live Gemini and writes a JSON report, so a before and after can be compared. One live run is noisy, so several are averaged.
- **The mutation test** checks the checker: it corrupts real explanations on purpose and counts what is caught, and counts false alarms on untouched ones.
- **Humans stay in the loop.** Only a person can judge whether an explanation is good, rather than merely consistent with the evidence. People also decide what to change.

## Where the AI is checked

| # | Check | Type | When |
|---|---|---|---|
| 1 | Guardrail (`validate_profile`, plus catalog-label check for parsed text) | Automatic | Every request |
| 2 | Faithfulness checker | Automatic | Every AI explanation |
| 3 | Confidence score and low-confidence warning | Automatic, shown to the user | Every request |
| 4 | Pytest suite | Automatic, offline | On demand |
| 5 | Evaluation harness and mutation test | Scripted, live Gemini | On demand |
| 6 | Human spot-check of explanations, notes and the run log | Manual | By the developer |

## What the checks cannot do

- The faithfulness checker is a heuristic. It raised false alarms on natural phrasing such as "close acoustic feel".
- The confidence score cannot tell a plausible-looking wrong result from a right one.
- Neither automatic check can judge whether an explanation is good writing, only whether it contradicts the evidence.

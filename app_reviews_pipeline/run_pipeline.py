
# ===========================
# Imports & Path Setup
# ===========================
import sys
from pathlib import Path
import traceback

# ===========================
# Pipeline Root & Submodule Import Paths
# ===========================
PIPELINE_ROOT = Path(__file__).resolve().parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
for sub in ["M1_Discrepancy", "M2_Absa_recommendation", "M3_Topic_modeling", "M4_Rag_qa"]:
    p = PIPELINE_ROOT / sub
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ===========================
# Module Entrypoints
# ===========================
from M1_Discrepancy.discrepancy import main as run_discrepancy   # type: ignore
from M2_Absa_recommendation.absa_recomendation import main as run_absa  # type: ignore
# from M3_Topic_modeling.topic_modeling import main as run_topic   # type: ignore
from M4_Rag_qa.rag_qa import main as run_rag                     # type: ignore

# ===========================
# Menu & User Interaction
# ===========================
MENU = """
    ========================================
    App Reviews Pipeline — Module Launcher
    ========================================
    1) M1 — Discrepancy Analysis (VADER baseline)
    2) M2 — ABSA + Recommendation Mining
    3) M3 — Topic Modeling (BERTopic + LLM labels)
    4) M4 — RAG QA (evidence-grounded)
    5) Run ALL (paired defaults: GP→M1, AWARE→M2, Spotify→M3→M4)
    q) Quit
    ----------------------------------------
    Choose an option: 

"""

# ===========================
# Module Runner
# ===========================
def run_module(choice: str):
    if choice == "1":
        print("\n→ Running M1: Discrepancy Analysis …\n")
        run_discrepancy()
    elif choice == "2":
        print("\n→ Running M2: ABSA + Recommendations …\n")
        run_absa()
    # elif choice == "3":
    #     print("\n→ Running M3: Topic Modeling …\n")
    #     run_topic()
    elif choice == "4":
        print("\n→ Running M4: RAG QA …\n")
        run_rag()
    elif choice == "5":
        print("\n→ Running ALL with suggested pairings …\n")
        run_discrepancy()
        run_absa()
        run_topic()
        run_rag()
    else:
        print("Unknown option.")

# ===========================
# Main Loop
# ===========================
def main():
    while True:
        try:
            choice = input(MENU).strip().lower()
            if choice in {"q", "quit", "exit"}:
                print("Good-bye!")
                break
            run_module(choice)
        except KeyboardInterrupt:
            print("\nInterrupted. Back to menu.")
        except Exception:
            print("\n⚠️ An error occurred while running the selected module:\n")
            traceback.print_exc()
            print("\nReturning to menu …\n")

if __name__ == "__main__":
    main()


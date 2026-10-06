from src.agent import InventoryAgent
from src.db import get_logs

if __name__ == "__main__":
    agent = InventoryAgent()
    print(f"Initial Agent State: {agent.state}")

    print("\nRunning Agent Cycle...")
    results = agent.run_agent_cycle(is_weekend=True, has_promo=False)

    print(f"Final Agent State: {agent.state}")
    print(f"Processed {len(results)} products.")

    print("\nSample Log Entry from SQLite:")
    df_logs = get_logs()
    print(df_logs.head(3))
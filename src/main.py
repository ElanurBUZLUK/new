import argparse

from langchain_core.messages import HumanMessage

from .agent import build_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", type=str)
    args = ap.parse_args()

    graph = build_graph()
    out = graph.invoke({"messages": [HumanMessage(content=args.prompt)]})
    print(out["messages"][-1].content)


if __name__ == "__main__":
    main()

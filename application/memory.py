from loguru import logger
from collections import defaultdict
from typing import DefaultDict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Item:
    order: int = 0
    calls: int = 0
    val: Optional[Any] = None


@dataclass
class controller:
    forms: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_form(self):
        pass




logger.info("Initialize MEMORY")
MEMORY: DefaultDict[str, int] = defaultdict(int)

memo: DefaultDict[str, Item] = defaultdict(Item)


if __name__ == "__main__":
    memo["form1"] = Item(0, 0, "a")
    print(memo)
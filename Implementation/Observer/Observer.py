class StatisticsObserver:
    def __init__(self):
        self.statistics_drawer = []

    def notify(self):
        for s in self.statistics_drawer:
            s.update()

class StatisticsDrawer:
    def __init__(self):
        self.statistics = []

    def update(self):
        pass
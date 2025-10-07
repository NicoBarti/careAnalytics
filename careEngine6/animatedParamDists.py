from MyClasses.pathFinder import PathFinder

class AnimatedParamDists(PathFinder):
    def __init__(self, working_directory, allRuns_csv, ENGINE_PATH, varsigma, selectionName, minH, maxH):
        super().__init__(working_directory=working_directory, allRuns_csv=allRuns_csv, ENGINE_PATH=ENGINE_PATH, varsigma=varsigma)
        self.selectionName = selectionName
        self.createSelection(name=self.selectionName, minH=minH, maxH=maxH, color=0.6)



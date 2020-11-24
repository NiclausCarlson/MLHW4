import math

import Letter
import Utils
import matplotlib.pyplot as plt


def getNgram(arr, n):
    return list(zip(*[arr[i:] for i in range(n)]))


def generateCountMap(ngrams):
    nmaps = {}  # n-грамма | сколько раза была в спаме | сколько раз была в нормальном письме
    spamCounter, hamCounter = 0, 0
    for ngram in ngrams:  # каждая n-грама - это Letter, обработанный как n-грамма
        tmpMap = {}
        tmpList = ngram.subject + ngram.letter
        if ngram.isSpam:
            spamCounter += 1
        else:
            hamCounter += 1
        for gramma in tmpList:
            if gramma not in tmpMap:
                if gramma in nmaps:
                    if ngram.isSpam:
                        nmaps[gramma] = (nmaps[gramma][0] + 1, nmaps[gramma][1])
                    else:
                        nmaps[gramma] = (nmaps[gramma][0], nmaps[gramma][1] + 1)
                else:
                    if ngram.isSpam:
                        nmaps[gramma] = (1, 0)
                    else:
                        nmaps[gramma] = (0, 1)
            tmpMap[gramma] = True
    # if spamCounter == 0 and hamCounter == 0:
    #     f = open("shit.txt", 'w')
    #     for ngram in ngrams:
    #         f.write(str(ngram.subject) + '\n')
    #         f.write(str(ngram.letter) + '\n')
    #         f.write(str(ngram.isSpam) + '\n')
    #     f.close()
    #     exit(1)
    return spamCounter, hamCounter, nmaps


def copyClassifier(classifierFrom, classifierTo, accuracy):
    classifierTo.spamQuantity = classifierFrom.spamQuantity
    classifierTo.hamQuantity = classifierFrom.hamQuantity
    classifierTo.distribution = classifierFrom.distribution.copy()
    classifierTo.corpusSize = classifierFrom.corpusSize
    classifierTo.nGramSize = classifierFrom.nGramSize
    classifierTo.alpha = classifierFrom.alpha
    classifierTo.accuracy = accuracy


def getLetterNgram(n, letter):
    subject = getNgram(letter.subject, n)
    text = getNgram(letter.letter, n)
    return subject, text


class Classifier:

    def __init__(self, spamCounter, hamCounter, distribution, nGramSize, alpha, accuracy):
        self.nGramSize = nGramSize
        self.alpha = alpha
        self.distribution = distribution
        self.spamQuantity = spamCounter
        self.hamQuantity = hamCounter
        self.corpusSize = sum([x[1][0] + x[1][1] for x in distribution.items()])
        self.accuracy = accuracy
        self.Q = 2  # spam or not to spam

    def getWordProbability(self, nGram, predictionType):
        quantity = self.distribution.get(nGram)
        if quantity is None:
            quantity = (0, 0)
        probability = 0
        if predictionType == "spam":
            probability = (quantity[0] + self.alpha) / (self.corpusSize + self.alpha * self.Q)
        elif predictionType == "ham":
            probability = (quantity[1] + self.alpha) / (self.corpusSize + self.alpha * self.Q)
        return probability

    def getLetterProbability(self, lettersNgram):
        # compute ham probability
        isSpam = math.log1p(self.spamQuantity / len(self.distribution))
        # compute span probability
        isHam = math.log1p(self.hamQuantity / len(self.distribution))
        for nGrama in lettersNgram:
            isSpam += math.log1p(self.getWordProbability(nGrama, "spam"))
            isHam += math.log1p(self.getWordProbability(nGrama, "ham"))
        return isSpam, isHam

    # метод-классификатор
    def classifier(self, lettersNgram):
        isSpam, isHam = self.getLetterProbability(lettersNgram)
        return "spam" if isSpam >= isHam else "ham"

    # метод-классификатор с аргументом penaltyForHam
    def advancedClassifier(self, letter, spamPenalty, hamPenalty):
        ngramLetter = getNgram(letter, self.nGramSize)
        isSpam = math.log1p(self.spamQuantity / len(self.distribution))
        isHam = math.log1p(self.hamQuantity / len(self.distribution))
        # compute ham probability
        # compute span probability
        for nGrama in ngramLetter:
            isSpam += spamPenalty * math.log1p(self.getWordProbability(nGrama, "spam"))
            isHam += hamPenalty * math.log1p(self.getWordProbability(nGrama, "ham"))
        return "spam" if isSpam >= isHam else "ham"

    def str(self):
        return 'Accuracy: ' + str(self.accuracy) + '\n' + \
               'Ngram size: ' + str(self.nGramSize) + '\n' + \
               'Alpha: ' + str(self.alpha) + '\n'


class Bayes:
    def __init__(self):
        self.MAX_NGRAM_LENGTH = 3
        self.ALPHA_PARAMETERS = [1e-3 * ((x + 1) ** 4) for x in range(10)]
        self.DATASETS = Utils.getParts()
        self.BLOCK_QUANTITY = 10
        self.classifiers = []
        self.bestClassifier = Classifier(0, 0, {}, None, None, 0)
        self.PENALTY_STEP = 0.2

    def countAccuracy(self, testingBlockIndex, classifier):
        count = 0
        for letter in self.DATASETS[testingBlockIndex]:
            type = "spam" if letter.isSpam else "ham"
            subject, text = getLetterNgram(classifier.nGramSize, letter)
            predicted = classifier.classifier(subject + text)
            if type == predicted:
                count += 1
        return 100 * (count / len(self.DATASETS[testingBlockIndex]))

    def createNgrams(self, n, indexes):
        ngragms = []  # делаем n-грамы для каждого письма
        for i in indexes:  # перебираем индексы обучающих блоков
            for letter in self.DATASETS[i]:
                subject, text = getLetterNgram(n, letter)
                ngragms.append(Letter.Letter(subject, text, letter.isSpam))
        return ngragms

    def bayes(self):
        for curNgramLength in range(self.MAX_NGRAM_LENGTH):
            for alpha in self.ALPHA_PARAMETERS:
                for testDataIndex in range(self.BLOCK_QUANTITY):
                    indexes = [i for i in range(self.BLOCK_QUANTITY) if i != testDataIndex]
                    nGrams = self.createNgrams(curNgramLength + 1, indexes)
                    spamCounter, hamCounter, countMaps = generateCountMap(nGrams)
                    classifier = Classifier(spamCounter, hamCounter, countMaps, curNgramLength + 1, alpha, 0)
                    accuracy = self.countAccuracy(testDataIndex, classifier)
                    # print(curNgramLength + 1, alpha, accuracy)
                    if self.bestClassifier.accuracy < accuracy:
                        copyClassifier(classifier, self.bestClassifier, accuracy)

        totalAccuracy = 0
        totalSize = 0
        for i in range(self.BLOCK_QUANTITY):
            totalSize += len(self.DATASETS[i])
            for letter in self.DATASETS[i]:
                type = "spam" if letter.isSpam else "ham"
                subject, text = getLetterNgram(self.bestClassifier.nGramSize, letter)
                predicted = self.bestClassifier.classifier(subject + text)
                if type == predicted:
                    totalAccuracy += 1
        totalAccuracy = totalAccuracy / totalSize * 100
        self.bestClassifier.accuracy = totalAccuracy

    def printResults(self):
        f = open("results.txt", "w")
        f.write(self.bestClassifier.str())
        f.close()

    class Point:
        def __init__(self, number, isSpam, isHam, predicted, type):
            self.number = number
            self.isSpam = isSpam
            self.isHam = isHam
            self.predicted = predicted
            self.type = type

    def plotRoc(self):
        idx = 0
        objects = []
        spam, ham = 0, 0
        for i in range(self.BLOCK_QUANTITY):
            for letter in self.DATASETS[i]:
                subject, text = getLetterNgram(self.bestClassifier.nGramSize, letter)
                isSpam, isHam = self.bestClassifier.getLetterProbability(subject + text)
                predicted = "spam" if isSpam > isHam else "ham"
                objects.append(self.Point(idx, isSpam, isHam, predicted, letter.isSpam))
                if letter.isSpam:
                    spam += 1
                else:
                    ham += 1
                idx += 1
        # objects.sort(key=lambda x: x.isHam if x.predicted == "ham" else (1 - x.isSpam)/spam,
        #              reverse=True)  # строю ROC по принадлежности письма к классу HAM
        objects.sort(key=lambda x: max(x.isHam, x.isSpam),
                     reverse=True)
        print(spam, ham)
        fpr, tpr, j = [0], [0], 0
        for i in range(spam + ham):
            if objects[i].type:  # spam == True
                fpr.append(fpr[j] + 1 / spam)
                tpr.append(tpr[j])
            else:
                fpr.append(fpr[j])
                tpr.append(tpr[j] + 1 / ham)
            j += 1

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set(title='ROC-кривая',
               xlabel='False positive',
               ylabel='True positive')

        ax.plot(fpr, tpr, color='red')
        plt.show()
        fig.savefig("ROC-кривая")

    def getHeuristic(self):
        penaltyForHam = 1
        penaltyStep = 0.02
        penaltyValues = []
        accuracyPoints = []
        isGoodHeuristic = False
        while not isGoodHeuristic:
            isAllGood = True
            penaltyValues.append(penaltyForHam)
            accuracy = 0
            for i in range(self.BLOCK_QUANTITY):
                for letter in self.DATASETS[i]:
                    type = "spam" if letter.isSpam else "ham"
                    subject, text = getLetterNgram(self.bestClassifier.nGramSize, letter)
                    predicted = self.bestClassifier.advancedClassifier(subject + text, 1, penaltyForHam)
                    if type == predicted:
                        accuracy += 1
                    elif type == "ham" and type != predicted:
                        isAllGood = False
                accuracyPoints.append(accuracy)
                if not isAllGood:
                    penaltyForHam += penaltyStep
                else:
                    isGoodHeuristic = True
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set(title='Зависимость от эвристики',
               xlabel='Penalty',
               ylabel='Accuracy')

        ax.plot(penaltyValues, accuracyPoints, color='red')
        plt.show()
        fig.savefig("Зависимость от эвристики")


b = Bayes()
b.bayes()
b.printResults()
b.plotRoc()
b.getHeuristic()

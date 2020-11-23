import math

import Letter
import Utils
import numpy as np
import matplotlib.pyplot as plt


def getNgram(arr, n):
    return list(zip(*[arr[i:] for i in range(n)]))


def concatGram(gramma):
    if len(gramma) == 1:
        return str(gramma[0])
    return ' '.join(map(str, gramma))


def generateCountMap(ngrams):
    nmaps = {}  # n-грамма | сколько раза было в спаме | сколько раз была в нормальном письме
    spamCounter, hamCounter = 0, 0
    for ngram in ngrams:  # каждая n-грама - это Letter, обработанный как n-грамма
        tmpMap = {}
        tmpList = ngram.subject + ngram.letter
        if ngram.isSpam:
            spamCounter += 1
        else:
            hamCounter += 1
        for gramma in tmpList:
            cGramma = concatGram(gramma)
            if cGramma not in tmpMap:
                if cGramma in nmaps:
                    if ngram.isSpam:
                        nmaps[cGramma] = (nmaps[cGramma][0] + 1, nmaps[cGramma][1])
                    else:
                        nmaps[cGramma] = (nmaps[cGramma][0], nmaps[cGramma][1] + 1)
                else:
                    if ngram.isSpam:
                        nmaps[cGramma] = (1, 0)
                    else:
                        nmaps[cGramma] = (0, 1)
            tmpMap[cGramma] = True

    return spamCounter, hamCounter, nmaps


def copyClassifier(classifierFrom, classifierTo, accuracy):
    classifierTo.distribution = classifierFrom.distribution
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
        self.corpusSize = sum([x[1][0] + x[1][1] for x in self.distribution.items()])
        self.accuracy = accuracy
        self.Q = 2  # spam or not to spam

    def getWordProbability(self, nGram, predictionType):
        quantity = self.distribution.get(concatGram(nGram))
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
        #  print(totalAccuracy, totalSize)
        self.bestClassifier.accuracy = totalAccuracy

    def printResults(self):
        f = open("results.txt", "w")
        f.write(self.bestClassifier.str())
        f.close()

    class Point:
        def __init__(self, number, isSpam, isHam, type):
            self.number = number
            self.isSpam = isSpam
            self.isHam = isHam
            self.type = type

    def plotRoc(self):
        idx = 0
        objects = []
        spamPointsSp = []
        spamPointsHm = []
        hamPointsSp = []
        hamPointsHm = []
        spam, ham = 0, 0
        for i in range(self.BLOCK_QUANTITY):
            for letter in self.DATASETS[i]:
                subject, text = getLetterNgram(self.bestClassifier.nGramSize, letter)
                isSpam, isHam = self.bestClassifier.getLetterProbability(subject + text)
                objects.append(self.Point(idx, math.exp(isSpam), math.exp(isHam), letter.isSpam))
                if letter.isSpam:
                    spamPointsSp.append(objects[idx].isSpam)
                    spamPointsHm.append(objects[idx].isHam)
                    spam += 1
                else:
                    hamPointsSp.append(objects[idx].isSpam)
                    hamPointsHm.append(objects[idx].isHam)
                    ham += 1
                idx += 1
        sorted(objects, key=lambda x: max(x.isSpam, x.isHam), reverse=True)
        fpr, tpr, j = [0], [0], 0
        for i in range(spam + ham):
            if objects[i].isSpam:
                fpr.append(fpr[j] + 1 / spam)
                tpr.append(tpr[j])
            else:
                fpr.append(fpr[j])
                tpr.append(tpr[j] + 1 / ham)
            j += 1

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(fpr, tpr)
        ax.scatter(spamPointsSp, spamPointsHm, color='g')
        ax.scatter(hamPointsSp, hamPointsHm, color='r')

    def getHeuristic(self):
        return 0


b = Bayes()
b.bayes()
b.printResults()
b.plotRoc()
b.getHeuristic()

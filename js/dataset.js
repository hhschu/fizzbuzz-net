import { readFileSync } from 'fs';
import { shuffle } from './shuffle.js';

function readCSV(fileName) {
    return readFileSync(fileName)
        .toString()
        .trim()
        .split('\n')
        .map(e => e.trim())
        .map(e => e.split(',').map(e => parseInt(e.trim())));
}

function oneHot(data, numClasses) {
    let oneHot = new Array(numClasses).fill(0);
    oneHot[data] = 1;
    return oneHot;
}

const trainData = readCSV('../data/train_data.csv')
const trainLabel = readCSV('../data/train_label.csv')
const testData = readCSV('../data/test_data.csv')
const testLabel = readCSV('../data/test_label.csv')

const trainDataset = shuffle(trainData.map((e, i) => ({ input: e, output: oneHot(trainLabel[i], 4) })))
const testDataset = testData.map((e, i) => ({ input: e, output: oneHot(testLabel[i], 4) }))

export { trainDataset, testDataset }

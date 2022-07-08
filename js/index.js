import { NeuralNetwork } from 'brain.js';
import { trainDataset, testDataset } from './dataset.js'
import { accuracy } from './eval.js';

const net = new NeuralNetwork(
    { hiddenLayers: [500], activation: 'relu' }
);

net.train(
    trainDataset,
    { iterations: 200, log: true, logPeriod: 10, learningRate: 0.01 }
);

const acc = accuracy(net, testDataset) * 100;
console.log(`Test Accuracy: ${acc.toFixed(2)}%`);

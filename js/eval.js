export function accuracy (net, dataset) {
    let hits = 0;
    for (const sample of dataset) {
        const prob = net.run(sample.input);
        let y = prob.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
        let yHat = sample.output.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
        if (y === yHat) {
            hits++;
        }
    }
    return hits / dataset.length;
}

/**
 * AeroLung Technical DSP Utility
 * Client-side signal processing for the Technical Dashboard.
 */

/**
 * Simulates quantization of a signal.
 * @param {number[]} data - The input signal.
 * @param {number} bits - Bit depth (4-24).
 * @returns {number[]} - Quantized signal.
 */
export const quantize = (data, bits) => {
  const levels = Math.pow(2, bits - 1) - 1;
  return data.map(v => Math.round(v * levels) / levels);
};

/**
 * Simple Downsampling (Decimation).
 * @param {number[]} data - The input signal.
 * @param {number} factor - Downsampling factor (1-8).
 * @returns {number[]} - Downsampled signal.
 */
export const downsample = (data, factor) => {
  if (factor <= 1) return data;
  return data.filter((_, i) => i % factor === 0);
};

/**
 * Basic FIR Filter (Moving Average as placeholder for FIR).
 * @param {number[]} data - The input signal.
 * @param {number} windowSize - Window size.
 * @returns {number[]} - Filtered signal.
 */
export const applyFIR = (data, windowSize = 5) => {
  const output = [];
  for (let i = 0; i < data.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - Math.floor(windowSize / 2)); j <= Math.min(data.length - 1, i + Math.floor(windowSize / 2)); j++) {
      sum += data[j];
      count++;
    }
    output.push(sum / count);
  }
  return output;
};

/**
 * Basic IIR Filter (Simple Low-pass).
 * @param {number[]} data - The input signal.
 * @param {number} alpha - Smoothing factor (0-1).
 * @returns {number[]} - Filtered signal.
 */
export const applyIIR = (data, alpha = 0.5) => {
  const output = [data[0]];
  for (let i = 1; i < data.length; i++) {
    output.push(alpha * data[i] + (1 - alpha) * output[i - 1]);
  }
  return output;
};

/**
 * Mock FFT Power Spectrum calculation.
 * (In a real scenario, use an FFT library or Web Audio API)
 */
export const calculatePowerSpectrum = (data) => {
  // Mocking spectral density based on signal frequency components
  // Real FFT would be needed for production.
  const bins = 100;
  const spectrum = [];
  for (let i = 0; i < bins; i++) {
    const freq = i * (4000 / bins);
    // Add some mock peaks
    const noise = Math.random() * 0.1;
    const peak1 = Math.exp(-Math.pow(freq - 500, 2) / 20000) * 0.6;
    const peak2 = Math.exp(-Math.pow(freq - 1500, 2) / 10000) * 0.4;
    spectrum.push({ freq, magnitude: peak1 + peak2 + noise });
  }
  return spectrum;
};

/**
 * Generates technical metrics for the diagnostic panel.
 */
export const generateDiagnosticMetrics = (status) => {
  const isWheeze = status.toLowerCase().includes('wheeze');
  const isCrackle = status.toLowerCase().includes('crackle');
  
  return {
    peakFrequency: isWheeze ? 850 + Math.random() * 200 : (isCrackle ? 120 + Math.random() * 50 : 250 + Math.random() * 100),
    latency: 12 + Math.random() * 5,
    confidence: 0.88 + Math.random() * 0.1
  };
};

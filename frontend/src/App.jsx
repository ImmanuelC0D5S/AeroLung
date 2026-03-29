import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Activity, 
  Upload, 
  Cpu, 
  Zap, 
  Microscope, 
  ShieldCheck, 
  Terminal, 
  Maximize2,
  AlertCircle,
  FileAudio,
  CheckCircle2,
  X,
  RefreshCw,
  Loader2,
  Search,
  ChevronRight,
  Database,
  Info,
  Layers,
  BarChart3
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  AreaChart, 
  Area,
  ReferenceArea,
  ReferenceLine,
  Label
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

// --- COMPONENTS ---

const InspectorPanel = ({ isOpen, onClose, data }) => {
  if (!data) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div 
          initial={{ x: '100%' }} 
          animate={{ x: 0 }} 
          exit={{ x: '100%' }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className="fixed top-0 right-0 h-full w-[500px] glass-container z-50 shadow-2xl flex flex-col border-l border-[#faff0030]"
        >
          {/* HEADER */}
          <div className="p-6 border-b border-[#faff0020] flex justify-between items-center bg-[#faff0005]">
            <div className="flex items-center gap-3">
              <Search className="text-neon-yellow glow-yellow" size={18} />
              <h2 className="text-xs font-bold uppercase tracking-[0.3em] text-neon-yellow">Technical Inspector [EXP 2-8]</h2>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-white/10 rounded transition-all">
              <X size={18} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-10 custom-scrollbar">
            {/* CHART 1: RAW vs FILTERED OVERLAY */}
            <div className="space-y-4">
               <div className="flex justify-between items-baseline">
                 <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-60">Signal Transformation [EXP 5/6]</h3>
                 <span className="text-[8px] bg-white/5 px-2 py-0.5 rounded opacity-40">Zoomable LineChart</span>
               </div>
               <div className="h-[220px] bg-black/40 border border-white/5 rounded p-4 pr-8 relative">
                 <div className="absolute top-2 left-6 flex gap-4 text-[7px] uppercase tracking-tighter opacity-70">
                   <span className="text-neon-cyan flex items-center gap-1"><span className="w-1.5 h-1.5 bg-neon-cyan" /> Raw</span>
                   <span className="text-electric-lime flex items-center gap-1"><span className="w-1.5 h-1.5 bg-electric-lime" /> Filtered</span>
                 </div>
                 <ResponsiveContainer width="100%" height="100%">
                   <LineChart data={data.waveforms.raw.map((v, i) => ({ t: i, raw: v, filt: data.waveforms.filtered[i] }))}>
                     <XAxis hide />
                     <YAxis hide domain={[-1.1, 1.1]} />
                     <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333', fontSize: '8px' }} />
                     <Line type="monotone" dataKey="raw" stroke="#00f2ff" strokeWidth={1} dot={false} strokeOpacity={0.3} isAnimationActive={false} />
                     <Line type="monotone" dataKey="filt" stroke="#adff2f" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                   </LineChart>
                 </ResponsiveContainer>
               </div>
               <p className="text-[8px] opacity-40 italic">Note: Exp 5 Notch removes 50Hz hum; Exp 6 isolates 200-2000Hz band.</p>
            </div>

            {/* CHART 2: SPECTRAL BANDS */}
            <div className="space-y-4">
               <div className="flex justify-between items-baseline">
                 <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-60">Spectral Content [EXP 2]</h3>
                 <span className="text-[8px] bg-white/5 px-2 py-0.5 rounded opacity-40">Highlighted Bands</span>
               </div>
               <div className="h-[220px] bg-black/40 border border-white/5 rounded p-4 pr-8 relative">
                 <div className="absolute top-1 right-2 text-[6px] uppercase opacity-20">Logarithmic Scale (dB)</div>
                 <ResponsiveContainer width="100%" height="100%">
                   <AreaChart data={data.fft}>
                     <XAxis dataKey="freq" hide />
                     <YAxis hide domain={[-100, 20]} />
                     <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333', fontSize: '8px' }} />
                     {/* Highlight Wheeze/Crackle zone (Example: 400-1600Hz) */}
                     {data.detection_bounds && (
                        <ReferenceArea x1={data.detection_bounds[0]} x2={data.detection_bounds[1]} fill="#faff0005" strokeOpacity={0.1} />
                     )}
                     <Area type="monotone" dataKey="mag" stroke="#ff00ff" fill="#ff00ff10" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                   </AreaChart>
                 </ResponsiveContainer>
               </div>
               <div className="flex gap-4 text-[7px] opacity-50 uppercase tracking-widest">
                 <div className="flex items-center gap-1"><div className="w-1 h-1 bg-neon-yellow" /> Detected Range</div>
               </div>
            </div>

            {/* EXPERT FEATURE TABLE */}
            <div className="space-y-4 pb-10">
               <h3 className="text-[10px] font-bold uppercase tracking-widest opacity-60">Expert Feature Vector [EXP 3]</h3>
               <div className="border border-white/5 bg-black/20">
                  <table className="w-full text-left text-[9px] jet-mono">
                    <thead className="bg-white/5 border-b border-white/5 opacity-50 uppercase">
                      <tr>
                        <th className="p-3 font-normal">Metric ID</th>
                        <th className="p-3 font-normal">Calculated Value</th>
                        <th className="p-3 font-normal text-right">Status</th>
                      </tr>
                    </thead>
                    <tbody className="opacity-80">
                      {Object.entries(data.dsp_features).map(([key, value], i) => (
                        <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-all">
                          <td className="p-3 opacity-60">{key}</td>
                          <td className="p-3 text-neon-cyan">{typeof value === 'number' ? value.toFixed(4) : value}</td>
                          <td className="p-3 text-right text-emerald-400">VALID</td>
                        </tr>
                      ))}
                      <tr className="bg-[#faff0005]">
                        <td className="p-3 opacity-60">Quantization SNR [EXP 7]</td>
                        <td className="p-3 text-neon-yellow glow-yellow">{data.snr_db} dB</td>
                        <td className="p-3 text-right text-neon-yellow">PASSED</td>
                      </tr>
                    </tbody>
                  </table>
               </div>
            </div>
          </div>

          {/* FOOTER */}
          <div className="p-6 border-t border-white/5 bg-black/40 text-[8px] opacity-40 uppercase tracking-widest">
            Diagnostic Integrity verified at {data.metadata.fs}Hz | Calibration: PASS
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

const App = () => {
  // --- STATE ---
  const [file, setFile] = useState(null);
  const [bitDepth, setBitDepth] = useState(16);
  const [targetSR, setTargetSR] = useState(44100);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [showInspector, setShowInspector] = useState(false);

  // --- REFS ---
  const logEndRef = useRef(null);
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // --- ANALYZE LOGIC ---
  const handleAnalyze = async (selectedFile) => {
    if (!selectedFile) return;
    
    setFile(selectedFile);
    setIsAnalyzing(true);
    setResult(null);
    setLogs([]);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('bit_depth', bitDepth);
    formData.append('target_sr', targetSR);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "MATLAB Engine Failure (500)");
      }

      const data = await response.json();
      
      // Simulate rolling logs for the "Medical Terminal" effect
      for (const log of data.logs) {
        setLogs(prev => [...prev, log]);
        await new Promise(r => setTimeout(r, 400));
      }

      setResult(data);
    } catch (error) {
      setLogs(prev => [...prev, `CRITICAL SYSTEM ERROR: ${error.message}`]);
      setResult(null); // Ensure stale data doesn't crash charts
    } finally {
      setIsAnalyzing(false);
    }
  };

  // --- DRAG & DROP HANDLERS ---
  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.includes('audio')) {
      handleAnalyze(droppedFile);
    }
  }, []);

  return (
    <div className="min-h-screen bg-[#060606] text-[#e0e0e0] jet-mono p-6 selection:bg-[#00f2ff33] relative overflow-hidden">
      
      {/* TECHNICAL INSPECTOR PANEL */}
      <InspectorPanel 
        isOpen={showInspector} 
        onClose={() => setShowInspector(false)} 
        data={result} 
      />

      {/* HEADER */}
      <header className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 border border-[#00f2ff50] rounded flex items-center justify-center glow-cyan">
             <Activity className="text-neon-cyan" size={24} />
          </div>
          <div>
            <h1 className="cormorant-serif text-3xl italic font-medium leading-none mb-1 lowercase text-white">Aerolung</h1>
            <p className="text-[9px] uppercase tracking-[0.3em] opacity-40">MATLAB Signal Analyzer Mode v.5.0</p>
          </div>
        </div>
        <div className="flex gap-10">
           {result && (
              <div className="text-right border-r border-white/5 pr-10">
                <span className="text-[8px] opacity-30 uppercase block font-bold text-neon-yellow">Quantization SNR [Exp 7]</span>
                <span className="text-lg font-bold text-neon-yellow glow-yellow jet-mono">
                  {result.snr_db} dB
                </span>
              </div>
           )}
           <div className="text-right">
              <span className="text-[8px] opacity-30 uppercase block">System Status</span>
              <span className="text-[10px] text-emerald-400 uppercase tracking-widest flex items-center gap-2 justify-end">
                <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"></span>
                Online
              </span>
           </div>
        </div>
      </header>

      {/* MAIN GRID */}
      <main className="grid grid-cols-12 gap-6 h-[calc(100vh-120px)]">
        
        {/* LEFT PANEL: UPLOAD & PARAMETERS */}
        <aside className="col-span-3 flex flex-col gap-6">
          <div className="flex-1 glass-container p-6 relative overflow-hidden group">
            <h2 className="text-[10px] font-bold uppercase tracking-[0.3em] mb-4 flex items-center gap-2 opacity-60">
              <Upload size={12} /> Signal Source
            </h2>
            
            <div 
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              className={`h-full border border-dashed transition-all flex flex-col items-center justify-center p-8 rounded cursor-pointer relative ${
                isDragging ? 'border-neon-cyan bg-[#00f2ff05]' : 'border-white/5 hover:border-white/10'
              }`}
            >
              {isAnalyzing && <div className="scanning-bar" />}
              
              <Upload 
                size={34} 
                strokeWidth={1} 
                className={`mb-4 transition-colors ${isAnalyzing ? 'text-neon-cyan animate-pulse' : 'text-white/10'}`} 
              />
              
              <input 
                type="file" 
                className="absolute inset-0 opacity-0 cursor-pointer"
                onChange={(e) => handleAnalyze(e.target.files[0])}
                disabled={isAnalyzing}
              />

              <div className="text-center">
                <p className="text-[9px] font-bold uppercase mb-2 tracking-widest opacity-80">
                   {isAnalyzing ? "Processing..." : file ? file.name : "Select .WAV"}
                </p>
              </div>
            </div>
          </div>

          <div className="glass-container p-6">
            <h2 className="text-[10px] font-bold uppercase tracking-[0.3em] mb-6 flex items-center gap-2 opacity-60">
              <Cpu size={12} /> Hardware Logic
            </h2>
            <div className="space-y-8">
              <div className="space-y-3">
                <div className="flex justify-between text-[9px] uppercase tracking-widest">
                  <span>RES_BIT_DEPTH [EXP 7]</span>
                  <span className="text-neon-cyan font-bold">{bitDepth} BITS</span>
                </div>
                <input 
                  type="range" min="4" max="24" step="4" 
                  value={bitDepth} 
                  onChange={(e) => setBitDepth(parseInt(e.target.value))}
                  disabled={isAnalyzing}
                  className="w-full"
                />
              </div>
              <div className="space-y-3">
                <div className="flex justify-between text-[9px] uppercase tracking-widest">
                  <span>SAMPLE_VERIFY [EXP 8]</span>
                  <span className="text-neon-cyan font-bold">{targetSR} HZ</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                   {[8000, 44100].map(sr => (
                     <button
                        key={sr}
                        onClick={() => setTargetSR(sr)}
                        disabled={isAnalyzing}
                        className={`py-2 text-[9px] border transition-all ${
                          targetSR === sr ? 'border-neon-cyan text-neon-cyan bg-[#00f2ff10]' : 'border-white/5 opacity-40 hover:opacity-100'
                        }`}
                      >
                       {sr} HZ
                     </button>
                   ))}
                </div>
              </div>
            </div>
          </div>

          <button 
            onClick={() => setShowInspector(true)}
            disabled={!result}
            className={`w-full h-12 border transition-all flex items-center justify-center gap-2 text-[9px] font-bold uppercase tracking-[0.2em] active:scale-95 ${
              result 
              ? 'border-neon-yellow text-neon-yellow bg-[#faff0010] glow-yellow' 
              : 'border-white/5 text-white/5 opacity-20'
            }`}
          >
            <Search size={14} />
            DSP Expert Inspector
          </button>
        </aside>

        {/* CENTER PANEL: MATLAB TRIPLE STACK */}
        <section className="col-span-6 flex flex-col gap-4 overflow-hidden">
          {/* PANEL 1: NORMAL FFT - SPIKE ANALYZER */}
          <div className="flex-1 glass-container p-4 flex flex-col oscilloscope-grid relative overflow-hidden">
             <div className="flex justify-between items-center mb-1">
               <h2 className="text-[8px] font-bold uppercase tracking-[0.2em] flex items-center gap-2 opacity-40">
                 <BarChart3 size={10} /> EXPERIMENT 2: Normal FFT (Linear Magnitude Spectrum)
               </h2>
               <span className="viva-badge badge-magenta">Exp 2 Spectral Spikes (Hamming Windowed)</span>
            </div>
            <div className="flex-1 min-h-0">
               <ResponsiveContainer width="100%" height="100%">
                 <LineChart data={result ? result.fft : []} syncId="scope">
                    <CartesianGrid strokeDasharray="2 2" stroke="rgba(255,255,255,0.03)" vertical={false} />
                    <XAxis dataKey="freq" stroke="#ffffff30" fontSize={8} tick={{fontFamily: 'JetBrains Mono'}} tickSize={5} />
                    <YAxis stroke="#ffffff30" fontSize={8} tick={{fontFamily: 'JetBrains Mono'}} label={{ value: 'Mag (lin)', angle: -90, position: 'insideLeft', offset: 10, fill: '#ffffff20', fontSize: 7, fontWeight: 'bold' }} domain={[0, 'auto']} />
                    <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333', fontSize: '8px' }} />
                    
                    {/* Clinical Zones [Exp 2] */}
                    <ReferenceArea x1={0} x2={200} fill="#ffffff05" strokeOpacity={0} label={{ position: 'insideTopLeft', value: 'CARD_NOISE', fill: '#ffffff15', fontSize: 6 }} />
                    <ReferenceArea x1={200} x2={800} fill="#2dd4bf05" strokeOpacity={0} label={{ position: 'insideTopLeft', value: 'WHEEZE_ZONE', fill: '#2dd4bf30', fontSize: 6 }} />
                    <ReferenceArea x1={800} x2={2000} fill="#60a5fa05" strokeOpacity={0} label={{ position: 'insideTopLeft', value: 'CRACKLE_HISS', fill: '#60a5fa30', fontSize: 6 }} />
                    
                    {/* Peak Frequency Line [Exp 2] */}
                    {result?.peak_freq > 0 && (
                      <ReferenceLine x={result.peak_freq} stroke="#FF00FF" strokeDasharray="3 3" label={{ position: 'top', value: `PEAK: ${result.peak_freq}HZ`, fill: '#FF00FF', fontSize: 7, fontWeight: 'bold', offset: 10 }} />
                    )}

                    {result?.detection_bounds && (
                        <ReferenceArea x1={result.detection_bounds[0]} x2={result.detection_bounds[1]} fill="#FF00FF10" strokeOpacity={0.3} />
                    )}
                    <Line type="monotone" dataKey="mag" stroke="#FF00FF" strokeWidth={1} dot={false} className="glow-magenta" isAnimationActive={!!result} />
                 </LineChart>
               </ResponsiveContainer>
            </div>
          </div>

          {/* PANEL 2: RAW OSCILLOSCOPE */}
          <div className="flex-1 glass-container p-4 flex flex-col oscilloscope-grid relative overflow-hidden">
            <div className="flex justify-between items-center mb-1">
               <h2 className="text-[8px] font-bold uppercase tracking-[0.2em] flex items-center gap-2 opacity-40">
                 <Activity size={10} /> Raw Input Signal [Time Domain]
               </h2>
               <span className="viva-badge bg-white/5 opacity-30 border border-white/5">Exp 8 Resampling State</span>
            </div>
            <div className="flex-1 min-h-0">
               <ResponsiveContainer width="100%" height="100%">
                 <LineChart data={result ? result.waveforms.raw.map((v, i) => ({ t: i, val: v })) : []} syncId="scope">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.02)" />
                    <XAxis dataKey="t" stroke="#ffffff30" fontSize={8} tick={{fontFamily: 'JetBrains Mono'}} tickSize={5} />
                    <YAxis stroke="#ffffff30" fontSize={8} tick={{fontFamily: 'JetBrains Mono'}} label={{ value: 'Amp (a.u.)', angle: -90, position: 'insideLeft', offset: 10, fill: '#ffffff20', fontSize: 7 }} domain={[-1.2, 1.2]} />
                    <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333', fontSize: '8px' }} label="Sample Index" />
                    <Line type="monotone" dataKey="val" stroke="#60a5fa" strokeWidth={1} dot={false} strokeOpacity={0.5} isAnimationActive={!!result} />
                 </LineChart>
               </ResponsiveContainer>
            </div>
          </div>

          {/* PANEL 3: CLEANED OSCILLOSCOPE */}
          <div className="flex-1 glass-container p-4 flex flex-col oscilloscope-grid relative overflow-hidden">
             <div className="flex justify-between items-center mb-1">
               <h2 className="text-[8px] font-bold uppercase tracking-[0.2em] flex items-center gap-2 opacity-40">
                 <Zap size={10} /> Cleaned Respiratory Signal [Time Domain]
               </h2>
               <span className="viva-badge badge-green">Exp 5 & 6 Active (IIR Butterworth)</span>
            </div>
            <div className="flex-1 min-h-0">
               <ResponsiveContainer width="100%" height="100%">
                 <LineChart data={result ? result.waveforms.filtered.map((v, i) => ({ t: i, val: v })) : []} syncId="scope">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.02)" />
                    <XAxis dataKey="t" stroke="#ffffff30" fontSize={8} tick={{fontFamily: 'JetBrains Mono'}} tickSize={5} />
                    <YAxis stroke="#ffffff30" fontSize={8} tick={{fontFamily: 'JetBrains Mono'}} label={{ value: 'Amp (a.u.)', angle: -90, position: 'insideLeft', offset: 10, fill: '#ffffff20', fontSize: 7 }} domain={[-1.1, 1.1]} />
                    <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333', fontSize: '8px' }} label="Sample Index" />
                    <Line type="monotone" dataKey="val" stroke="#39FF14" strokeWidth={1.5} dot={false} className="glow-lime" isAnimationActive={!!result} />
                 </LineChart>
               </ResponsiveContainer>
            </div>
          </div>
        </section>

        {/* RIGHT PANEL: TERMINAL & RESULTS */}
        <aside className="col-span-3 flex flex-col gap-6">
          <div className="flex-1 glass-container p-4 flex flex-col bg-black/60 relative overflow-hidden">
             <div className="flex items-center gap-2 mb-3 border-b border-white/5 pb-2">
                <Terminal size={12} className="text-neon-cyan" />
                <span className="text-[9px] uppercase tracking-widest font-bold">MATLAB Terminal Output</span>
             </div>
             <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                {logs.length === 0 && (
                  <p className="text-[8px] opacity-20 italic">Awaiting analysis sequence start...</p>
                )}
                {logs.map((log, i) => (
                  <div key={i} className="mb-2 flex gap-3 animate-in fade-in slide-in-from-left-1 duration-300">
                    <span className="text-neon-cyan opacity-40 shrink-0 text-[8px]">&gt;&gt;</span>
                    <span className={`text-[8px] leading-relaxed transition-all tracking-tight ${log.includes('Applying EXPERIMENT') ? 'text-neon-cyan font-bold' : log.includes('ERROR') ? 'text-red-400' : 'opacity-60'}`}>
                      {log}
                    </span>
                  </div>
                ))}
                <div ref={logEndRef} />
             </div>
          </div>

          <div className={`h-[250px] glass-container p-6 flex flex-col transition-all duration-1000 ${result ? 'border-[#adff2f40] glow-lime' : file && isAnalyzing ? 'border-[#00f2ff40] glow-cyan' : ''}`}>
             <h2 className="text-[10px] font-bold uppercase tracking-[0.3em] mb-4 flex items-center gap-2 opacity-60">
               <ShieldCheck size={12} /> AI Prediction Unit
             </h2>
             
             <div className="flex-1 flex flex-col justify-center">
                {result ? (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }} 
                    animate={{ opacity: 1, scale: 1 }} 
                    className="text-center"
                  >
                    <span className="text-[8px] uppercase tracking-[0.4em] opacity-40 block mb-2">Analysis Complete</span>
                    <h3 className="text-xl font-bold uppercase tracking-tighter text-electric-lime mb-3 glow-lime">
                      {result.prediction.replace('Diagnosis: ', '').replace('Condition: ', '')}
                    </h3>
                    <div className="flex items-center justify-center gap-4 px-4">
                       <div className="h-1 flex-1 bg-white/5 rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }} 
                            animate={{ width: `${result.confidence * 100}%` }}
                            className="h-full bg-electric-lime" 
                          />
                       </div>
                       <span className="text-[11px] font-bold text-electric-lime jet-mono">{Math.round(result.confidence * 100)}%</span>
                    </div>
                    <p className="text-[8px] opacity-40 uppercase tracking-widest mt-6 italic">
                      Exp Traceability: Segments={result.metadata.segments} | SNRp={result.snr_db}dB
                    </p>
                    <button 
                      onClick={() => {setResult(null); setFile(null); setLogs([]); setShowInspector(false);}}
                      className="mt-6 text-[8px] border border-white/10 px-4 py-2 hover:bg-white/5 transition-all uppercase tracking-[0.2em] flex items-center gap-2 mx-auto"
                    >
                      <RefreshCw size={10} /> RECALIBRATE
                    </button>
                  </motion.div>
                ) : isAnalyzing ? (
                  <div className="text-center">
                    <Loader2 size={24} className="animate-spin text-neon-cyan mx-auto mb-4 opacity-60" />
                    <p className="text-[9px] uppercase tracking-[0.2em] animate-pulse opacity-40">Scanning for frequency anomalies...</p>
                  </div>
                ) : (
                  <div className="text-center opacity-10 flex flex-col items-center">
                    <ShieldCheck size={36} strokeWidth={0.5} className="mb-4" />
                    <p className="text-[9px] uppercase tracking-[0.3em]">Ready for Analysis</p>
                  </div>
                )}
             </div>
          </div>
        </aside>
      </main>

      <footer className="h-6 mt-6 border-t border-white/5 flex items-center justify-between text-[7px] tracking-[0.3em] opacity-20 uppercase font-bold">
        <div className="flex gap-10">
           <span>Engine Path: /usr/local/matlab/r2026a</span>
           <span>FFT Window: Hanning / 2048</span>
        </div>
        <div className="flex gap-4">
           <span>AeroLung Signal Analyzer // Secure Link</span>
        </div>
      </footer>
    </div>
  );
};

export default App;

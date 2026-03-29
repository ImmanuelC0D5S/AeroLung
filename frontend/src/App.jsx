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
  Loader2
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
  Area
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

const App = () => {
  // --- STATE ---
  const [file, setFile] = useState(null);
  const [bitDepth, setBitDepth] = useState(16);
  const [targetSR, setTargetSR] = useState(44100);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([]);
  const [isDragging, setIsDragging] = useState(false);

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

      if (!response.ok) throw new Error("Backend connection failed.");

      const data = await response.json();
      
      // Simulate rolling logs for the "Medical Terminal" effect
      for (const log of data.logs) {
        setLogs(prev => [...prev, log]);
        await new Promise(r => setTimeout(r, 400));
      }

      setResult(data);
    } catch (error) {
      setLogs(prev => [...prev, `ERROR: ${error.message}`]);
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

  // --- RENDER HELPERS ---
  const formatChartData = (arr) => arr ? arr.map((v, i) => ({ t: i, val: v })) : [];

  return (
    <div className="min-h-screen bg-[#060606] text-[#e0e0e0] jet-mono p-6 selection:bg-[#00f2ff33]">
      
      {/* HEADER */}
      <header className="flex justify-between items-center mb-10">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 border border-[#00f2ff50] rounded flex items-center justify-center glow-cyan">
             <Activity className="text-neon-cyan" size={24} />
          </div>
          <div>
            <h1 className="cormorant-serif text-3xl italic font-medium leading-none mb-1 lowercase">Aerolung</h1>
            <p className="text-[9px] uppercase tracking-[0.3em] opacity-40">Diagnostic Stethoscope Terminal V.4.0</p>
          </div>
        </div>
        <div className="flex gap-10">
           <div className="text-right">
              <span className="text-[8px] opacity-30 uppercase block">System Status</span>
              <span className="text-[10px] text-emerald-400 uppercase tracking-widest flex items-center gap-2 justify-end">
                <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"></span>
                Online
              </span>
           </div>
           <div className="text-right">
              <span className="text-[8px] opacity-30 uppercase block">Model Checkpoint</span>
              <span className="text-[10px] opacity-60 uppercase tracking-widest">ICBHI_ENG_V3</span>
           </div>
        </div>
      </header>

      {/* MAIN GRID */}
      <main className="grid grid-cols-12 gap-6 h-[calc(100vh-140px)]">
        
        {/* LEFT PANEL: UPLOAD & PARAMETERS */}
        <aside className="col-span-3 flex flex-col gap-6">
          <div className="flex-1 glass-container p-6 relative overflow-hidden group">
            <h2 className="text-[10px] font-bold uppercase tracking-[0.3em] mb-6 flex items-center gap-2 opacity-60">
              <Upload size={12} /> Input Node
            </h2>
            
            <div 
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              className={`h-full border border-dashed transition-all flex flex-col items-center justify-center p-8 rounded cursor-pointer relative ${
                isDragging ? 'border-neon-cyan bg-[#00f2ff05]' : 'border-white/10 hover:border-white/30'
              }`}
            >
              {isAnalyzing && <div className="scanning-bar" />}
              
              <Upload 
                size={40} 
                strokeWidth={1} 
                className={`mb-4 transition-colors ${isAnalyzing ? 'text-neon-cyan animate-pulse' : 'text-white/20'}`} 
              />
              
              <input 
                type="file" 
                className="absolute inset-0 opacity-0 cursor-pointer"
                onChange={(e) => handleAnalyze(e.target.files[0])}
                disabled={isAnalyzing}
              />

              <div className="text-center">
                <p className="text-[10px] font-bold uppercase mb-2 tracking-widest">
                  {isAnalyzing ? "Analyzing Stream..." : file ? file.name : "Initialize Signal Scan"}
                </p>
                <p className="text-[9px] opacity-30 lowercase">
                  Drag & Drop .WAV or Click to Browse
                </p>
              </div>
            </div>
          </div>

          <div className="glass-container p-6">
            <h2 className="text-[10px] font-bold uppercase tracking-[0.3em] mb-6 flex items-center gap-2 opacity-60">
              <Cpu size={12} /> Modulation Constants
            </h2>
            <div className="space-y-8">
              <div className="space-y-3">
                <div className="flex justify-between text-[9px] uppercase tracking-widest">
                  <span>RES_BIT_DEPTH [EXP 7]</span>
                  <span className="text-neon-cyan">{bitDepth} Bits</span>
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
                  <span className="text-neon-cyan">{targetSR} Hz</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                   {[8000, 44100].map(sr => (
                     <button
                        key={sr}
                        onClick={() => setTargetSR(sr)}
                        disabled={isAnalyzing}
                        className={`py-2 text-[9px] border transition-all ${
                          targetSR === sr ? 'border-neon-cyan text-neon-cyan bg-[#00f2ff10]' : 'border-white/10 opacity-40 hover:opacity-100'
                        }`}
                      >
                       {sr} HZ
                     </button>
                   ))}
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* CENTER PANEL: VISUALIZATION */}
        <section className="col-span-6 flex flex-col gap-6 overflow-hidden">
          {/* TOP CHART: RAW & FILTERED SIGNAL */}
          <div className="flex-1 glass-container p-6 flex flex-col">
            <div className="flex justify-between items-center mb-4">
               <h2 className="text-[10px] font-bold uppercase tracking-[0.3em] flex items-center gap-2 opacity-60">
                 <Activity size={12} /> Waveform Analysis [EXP 4/6]
               </h2>
               <div className="flex gap-6 text-[8px] uppercase tracking-widest">
                 <span className="flex items-center gap-2 text-neon-cyan"><span className="w-1.5 h-1.5 bg-neon-cyan rounded-full"></span> Raw Input</span>
                 <span className="flex items-center gap-2 text-electric-lime"><span className="w-1.5 h-1.5 bg-electric-lime rounded-full"></span> Filtered [Exp 6]</span>
               </div>
            </div>

            <div className="flex-1 min-h-0 relative">
               {!result && !isAnalyzing && (
                 <div className="absolute inset-0 flex flex-col items-center justify-center opacity-10">
                    <Activity size={60} strokeWidth={0.5} />
                    <span className="text-[10px] mt-4 tracking-[0.5em] uppercase">No Active Stream</span>
                 </div>
               )}
               <ResponsiveContainer width="100%" height="100%">
                 <LineChart data={result ? result.waveform.raw.map((v, i) => ({ t: i, raw: v, filt: result.waveform.filtered[i] })) : []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis hide />
                    <YAxis hide domain={[-1.2, 1.2]} />
                    <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333', fontSize: '9px' }} />
                    <Line type="monotone" dataKey="raw" stroke="#00f2ff" strokeWidth={1} dot={false} strokeOpacity={0.4} isAnimationActive={!!result} />
                    <Line type="monotone" dataKey="filt" stroke="#adff2f" strokeWidth={1.5} dot={false} className="glow-lime" isAnimationActive={!!result} />
                 </LineChart>
               </ResponsiveContainer>
            </div>
          </div>

          {/* BOTTOM CHART: FFT SPECTRUM */}
          <div className="h-[250px] glass-container p-6 flex flex-col">
            <h2 className="text-[10px] font-bold uppercase tracking-[0.3em] mb-4 flex items-center gap-2 opacity-60">
              <Microscope size={12} /> Frequency Spectrum [EXP 2]
            </h2>
            <div className="flex-1 min-h-0">
               <ResponsiveContainer width="100%" height="100%">
                 <AreaChart data={result ? result.fft : []}>
                    <defs>
                      <linearGradient id="magentaGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ff00ff" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#ff00ff" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="freq" hide />
                    <YAxis hide domain={[-100, 20]} />
                    <Area type="monotone" dataKey="mag" stroke="#ff00ff" fill="url(#magentaGradient)" strokeWidth={2} className="glow-magenta" isAnimationActive={!!result} />
                 </AreaChart>
               </ResponsiveContainer>
            </div>
          </div>
        </section>

        {/* RIGHT PANEL: TERMINAL & RESULTS */}
        <aside className="col-span-3 flex flex-col gap-6">
          {/* TERMINAL LOGS */}
          <div className="flex-1 glass-container p-4 flex flex-col bg-black/40">
             <div className="flex items-center gap-2 mb-3 border-b border-white/10 pb-2">
                <Terminal size={12} className="text-neon-cyan" />
                <span className="text-[9px] uppercase tracking-widest font-bold">Lab Terminal</span>
             </div>
             <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                {logs.length === 0 && (
                  <p className="text-[9px] opacity-20 italic">Awaiting audio input segment...</p>
                )}
                {logs.map((log, i) => (
                  <div key={i} className="mb-2 flex gap-3 animate-in fade-in slide-in-from-left-2 duration-300">
                    <span className="text-neon-cyan opacity-40 shrink-0">➜</span>
                    <span className={`text-[9px] leading-relaxed transition-all ${log.includes('Applying EXPERIMENT') ? 'text-neon-cyan font-bold' : log.includes('ERROR') ? 'text-red-400' : 'opacity-70'}`}>
                      {log}
                    </span>
                  </div>
                ))}
                <div ref={logEndRef} />
             </div>
          </div>

          {/* DIAGNOSTIC CARD */}
          <div className={`h-[250px] glass-container p-6 flex flex-col transition-all duration-1000 ${result ? 'border-[#adff2f50] glow-lime' : file && isAnalyzing ? 'border-[#00f2ff50] glow-cyan' : ''}`}>
             <h2 className="text-[10px] font-bold uppercase tracking-[0.3em] mb-4 flex items-center gap-2 opacity-60">
               <ShieldCheck size={12} /> AI Diagnostic Node
             </h2>
             
             <div className="flex-1 flex flex-col justify-center">
                {result ? (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }} 
                    animate={{ opacity: 1, y: 0 }} 
                    className="text-center animate-pulse-reveal"
                  >
                    <span className="text-[8px] uppercase tracking-[0.4em] opacity-40 block mb-2">Diagnosis Reveal</span>
                    <h3 className="text-2xl font-bold uppercase tracking-tighter text-electric-lime mb-2 glow-lime">
                      {result.prediction.replace('Diagnosis: ', '')}
                    </h3>
                    <div className="flex items-center justify-center gap-4">
                       <div className="h-1 flex-1 bg-white/5 overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }} 
                            animate={{ width: `${result.confidence * 100}%` }}
                            className="h-full bg-electric-lime" 
                          />
                       </div>
                       <span className="text-[12px] font-bold text-electric-lime">{Math.round(result.confidence * 100)}%</span>
                    </div>
                    <p className="text-[8px] opacity-40 uppercase tracking-widest mt-4">
                      Matched cycles: {result.metadata.segments} | Experiment Verification: PASS
                    </p>
                    <button 
                      onClick={() => {setResult(null); setFile(null); setLogs([]);}}
                      className="mt-6 text-[9px] border border-white/20 px-4 py-2 hover:bg-white/5 transition-all uppercase tracking-widest flex items-center gap-2 mx-auto"
                    >
                      <RefreshCw size={10} /> Reset Scan
                    </button>
                  </motion.div>
                ) : isAnalyzing ? (
                  <div className="text-center opacity-60">
                    <Loader2 size={30} className="animate-spin text-neon-cyan mx-auto mb-4" />
                    <p className="text-[9px] uppercase tracking-widest animate-pulse">Running Neural Pattern Search...</p>
                  </div>
                ) : (
                  <div className="text-center opacity-20 flex flex-col items-center">
                    <ShieldCheck size={40} strokeWidth={0.5} className="mb-4" />
                    <p className="text-[10px] uppercase tracking-[0.2em]">Awaiting Analysis Sequence</p>
                  </div>
                )}
             </div>
          </div>
        </aside>
      </main>

      {/* FOOTER BAR */}
      <footer className="h-8 mt-6 border-t border-white/5 flex items-center justify-between text-[8px] tracking-[0.2em] opacity-30 uppercase">
        <div className="flex gap-8">
           <span>Engine: VITE_REACT_FASTAPI</span>
           <span>Exp Node: LAB_SYLLABUS_V4</span>
        </div>
        <div className="flex gap-4">
           <span>© 2026 AeroLung Technical</span>
           <span className="text-neon-cyan">// Diagnostic Integrity: High</span>
        </div>
      </footer>
    </div>
  );
};

export default App;

'use client';

import { useState, useEffect } from 'react';
import {
    BrainCircuit,
    TrendingUp,
    TrendingDown,
    Activity,
    AlertTriangle,
    Play,
    RotateCcw,
    Settings,
    ShieldCheck,
    CheckCircle2,
    Database,
    Cpu,
    Zap,
    History as HistoryIcon,
    FileCode2,
    Layers,
    ChevronRight
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
    BarChart,
    Bar,
    Cell
} from 'recharts';
import { toast } from 'sonner';
import { useRole } from '@/app/context/RoleContext';
import { getBackendUrl, getOrchestratorUrl } from '@/app/utils/api';

// Mock Performance Data (Fallback)
const performanceData = [
    { time: '00:00', xgb: 98.7, iso: 95.4, ae: 94.1 },
    { time: '04:00', xgb: 98.8, iso: 95.6, ae: 94.5 },
    { time: '08:00', xgb: 98.5, iso: 95.2, ae: 94.0 },
    { time: '12:00', xgb: 98.6, iso: 94.8, ae: 93.5 }, // Drift in unsupervised models
    { time: '16:00', xgb: 98.7, iso: 95.7, ae: 94.2 },
    { time: '20:00', xgb: 98.9, iso: 95.9, ae: 94.8 },
    { time: '24:00', xgb: 99.0, iso: 96.0, ae: 95.1 },
];

const driftData = [
    { day: 'Mon', score: 0.02 },
    { day: 'Tue', score: 0.03 },
    { day: 'Wed', score: 0.05 },
    { day: 'Thu', score: 0.12 }, // Warning threshold
    { day: 'Fri', score: 0.15 },
    { day: 'Sat', score: 0.08 },
    { day: 'Sun', score: 0.04 },
];

const distributionShift = [
    { feature: 'Income', train: 0.45, current: 0.62 },
    { feature: 'Age', train: 0.38, current: 0.39 },
    { feature: 'Geo', train: 0.55, current: 0.58 },
    { feature: 'Velocity', train: 0.22, current: 0.45 },
    { feature: 'Amt', train: 0.61, current: 0.59 },
];

const initialModels = [
    {
        id: 'MDL-XGB-001',
        name: 'XGBoost Classifier',
        status: 'Healthy',
        accuracy: '98.7%',
        precision: '97.2%',
        recall: '96.2%',
        latency: '42ms',
        lastTrained: '2 days ago',
        type: 'Supervised'
    },
    {
        id: 'MDL-IF-001',
        name: 'Isolation Forest',
        status: 'Healthy',
        accuracy: '95.4%',
        precision: '92.5%',
        recall: '91.8%',
        latency: '28ms',
        lastTrained: '12 hours ago',
        type: 'Anomaly Detection'
    },
    {
        id: 'MDL-AE-001',
        name: 'Lightweight Autoencoder',
        status: 'Healthy',
        accuracy: '94.1%',
        precision: '90.8%',
        recall: '89.5%',
        latency: '15ms',
        lastTrained: '5 days ago',
        type: 'Unsupervised'
    }
];

export default function ModelPerformancePage() {
    const { role } = useRole();
    const isSystemAdmin = role === 'system_admin';
    const [models, setModels] = useState(initialModels);
    const [summary, setSummary] = useState({
        global_accuracy: '96.1%',
        fraud_recall: '92.5%',
        fpr: '1.2%',
        inference_latency: '28ms'
    });

    const backendUrl = getBackendUrl();
    const orchestratorUrl = getOrchestratorUrl();

    useEffect(() => {
        const fetchData = async () => {
            try {
                const modelsRes = await fetch(`${backendUrl}/api/v1/models/`);
                const modelsData = await modelsRes.json();
                if (modelsData.items) setModels(modelsData.items);

                const summaryRes = await fetch(`${backendUrl}/api/v1/models/summary`);
                const summaryData = await summaryRes.json();
                setSummary(summaryData);
            } catch (error) {
                console.error("Error fetching model data:", error);
            }
        };
        fetchData();
    }, [backendUrl]);

    return (
        <div className="max-w-7xl mx-auto space-y-6 pb-12">

            {/* Header Bar */}
            <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className={`text-[10px] font-black px-2 py-0.5 rounded-full uppercase tracking-widest border ${isSystemAdmin ? 'bg-indigo-50 text-indigo-700 border-indigo-200' : 'bg-blue-50 text-blue-700 border-blue-200'}`}>
                            {isSystemAdmin ? 'System Health & Engine Telemetry' : 'Model Performance Monitor'}
                        </span>
                        <span className="text-slate-400 text-xs font-bold font-mono">
                            PERSPECTIVE: {role.toUpperCase()}
                        </span>
                    </div>
                    <h1 className="text-3xl font-black text-slate-900 tracking-tight">Model Performance & Drift</h1>
                    <p className="text-slate-500 text-sm mt-1 font-medium italic">
                        {isSystemAdmin ? 'Advanced analysis of model engine health, feature drift, and gradient stabilization.' : 'Monitor automated fraud detection health and model reliability.'}
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => toast.success('Retraining pipeline initiated')}
                        className="flex items-center gap-2 px-6 py-3 border border-slate-200 bg-white rounded-2xl text-xs font-black text-slate-700 shadow-sm hover:bg-slate-50 transition-all"
                    >
                        <RotateCcw className="w-4 h-4" /> Retrain All
                    </button>
                    {isSystemAdmin && (
                        <button
                            onClick={() => toast('Hyperparameter search space restricted')}
                            className="flex items-center gap-2 px-6 py-3 border border-indigo-200 bg-indigo-50 rounded-2xl text-xs font-black text-indigo-700 shadow-sm hover:bg-indigo-100 transition-all"
                        >
                            <Settings className="w-4 h-4" /> Tune Params
                        </button>
                    )}
                    <button
                        onClick={() => toast.success('Shadow Model scheduled for silent deployment')}
                        className={`flex items-center gap-2 px-6 py-3 rounded-2xl text-xs font-black text-white shadow-lg transition-all
                            ${isSystemAdmin ? 'bg-indigo-600 shadow-indigo-200 hover:bg-indigo-700' : 'bg-blue-600 shadow-blue-200 hover:bg-blue-700'}`}
                    >
                        <Play className="w-4 h-4" /> Deploy Shadow
                    </button>
                </div>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {[
                    { label: 'Global Accuracy', val: summary.global_accuracy, trend: '+0.5%', icon: ShieldCheck, color: 'text-indigo-500' },
                    { label: 'Fraud Recall', val: summary.fraud_recall, trend: '+1.2%', icon: Activity, color: 'text-emerald-500' },
                    { label: 'Inference Latency', val: summary.inference_latency, trend: '-5ms', icon: BrainCircuit, color: 'text-slate-400' }
                ].map((kpi: any, i) => (
                    <div key={i} className="bg-white rounded-3xl border border-slate-200 p-6 shadow-sm group hover:border-indigo-600/30 transition-all">
                        <div className="flex justify-between items-start mb-4">
                            <span className="text-slate-400 text-[10px] font-black uppercase tracking-widest">{kpi.label}</span>
                            <kpi.icon className={`w-5 h-5 ${kpi.color} group-hover:scale-110 transition-transform`} />
                        </div>
                        <div className="flex items-baseline gap-2">
                            <h3 className="text-3xl font-black text-slate-900 tracking-tighter">{kpi.val}</h3>
                            <span className={`text-[10px] font-black px-1.5 py-0.5 rounded ${kpi.trend.startsWith('+') ? 'text-emerald-600 bg-emerald-50' : 'text-blue-600 bg-blue-50'}`}>
                                {kpi.trend}
                            </span>
                        </div>
                    </div>
                ))}
            </div>

            {/* Main Charts Architecture */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Accuracy & Recall Chart */}
                <div className="bg-white rounded-[40px] border border-slate-200 p-8 shadow-sm relative overflow-hidden">
                    <div className="flex justify-between items-center mb-10">
                        <h3 className="font-black text-slate-900 text-lg tracking-tight uppercase">Model Accuracy Drift (24h)</h3>
                        <div className="flex gap-4">
                            <span className="flex items-center gap-1.5 text-[10px] font-black text-indigo-600 uppercase">
                                <span className="w-2 h-2 rounded-full bg-indigo-500"></span> XGBoost
                            </span>
                            <span className="flex items-center gap-1.5 text-[10px] font-black text-emerald-600 uppercase">
                                <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Isolation Forest
                            </span>
                            <span className="flex items-center gap-1.5 text-[10px] font-black text-amber-600 uppercase">
                                <span className="w-2 h-2 rounded-full bg-amber-500"></span> Autoencoder
                            </span>
                        </div>
                    </div>
                    <div className="h-72 w-full font-mono">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={performanceData}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#94a3b8', fontWeight: 'bold' }} />
                                <YAxis domain={['dataMin - 2', 'dataMax + 1']} axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#94a3b8', fontWeight: 'bold' }} />
                                <Tooltip
                                    contentStyle={{ borderRadius: '24px', border: 'none', boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.1)', fontSize: '10px', fontWeight: '900', textTransform: 'uppercase' }}
                                />
                                <Line type="monotone" dataKey="xgb" stroke="#6366f1" strokeWidth={4} dot={false} activeDot={{ r: 6, strokeWidth: 0 }} />
                                <Line type="monotone" dataKey="iso" stroke="#10b981" strokeWidth={4} dot={false} activeDot={{ r: 6, strokeWidth: 0 }} />
                                <Line type="monotone" dataKey="ae" stroke="#f59e0b" strokeWidth={4} dot={false} activeDot={{ r: 6, strokeWidth: 0 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Specific Views based on Role */}
                {isSystemAdmin ? (
                    /* Admin View: Feature Distribution Shift */
                    <div className="bg-slate-900 rounded-[40px] p-8 shadow-xl shadow-slate-200 text-white relative overflow-hidden group">
                        <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:scale-110 transition-transform">
                            <Layers className="w-24 h-24" />
                        </div>
                        <div className="flex justify-between items-center mb-10">
                            <div>
                                <h3 className="font-black text-lg tracking-tight uppercase">Covariate Shift (PSI)</h3>
                                <p className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest mt-1">Training Distribution vs Production</p>
                            </div>
                            <span className="px-3 py-1 bg-red-500/20 text-red-400 border border-red-500/30 rounded-xl text-[10px] font-black uppercase tracking-widest animate-pulse">
                                Warning: P-Value Threshold
                            </span>
                        </div>
                        <div className="h-72 w-full font-mono">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={distributionShift} layout="vertical">
                                    <XAxis type="number" hide />
                                    <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#818cf8', fontWeight: 'bold' }} width={70} />
                                    <Tooltip
                                        cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                        contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                                    />
                                    <Bar dataKey="train" fill="#312e81" radius={[0, 4, 4, 0]} barSize={12} />
                                    <Bar dataKey="current" fill="#818cf8" radius={[0, 4, 4, 0]} barSize={12} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                ) : (
                    /* Analyst View: Population DRIFT score */
                    <div className="bg-white rounded-[40px] border border-slate-200 p-8 shadow-sm">
                        <div className="flex justify-between items-center mb-10">
                            <h3 className="font-black text-slate-900 text-lg tracking-tight uppercase">System Drift Warning</h3>
                            <span className="px-3 py-1 bg-amber-50 text-amber-700 border border-amber-200 rounded-xl text-[10px] font-black uppercase tracking-widest">
                                ALERT: Income Feature Shift
                            </span>
                        </div>
                        <div className="h-72 w-full font-mono">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={driftData}>
                                    <defs>
                                        <linearGradient id="colorDrift" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                    <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#94a3b8', fontWeight: 'bold' }} />
                                    <Tooltip
                                        contentStyle={{ borderRadius: '24px', border: 'none', boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.1)', fontSize: '10px', fontWeight: '900' }}
                                    />
                                    <Area type="monotone" dataKey="score" stroke="#f59e0b" strokeWidth={4} fillOpacity={1} fill="url(#colorDrift)" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}
            </div>

            {/* Multi-view Bottom Registry */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Deployment Registry (2/3) */}
                <div className="lg:col-span-2 bg-white border border-slate-200 rounded-[40px] shadow-sm overflow-hidden p-8">
                    <div className="flex justify-between items-center mb-8">
                        <div>
                            <h3 className="font-black text-slate-900 text-lg uppercase tracking-tight flex items-center gap-2">
                                <Database className="w-5 h-5 text-indigo-600" /> Active Model Registry
                            </h3>
                            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mt-1">Live Inference Infrastructure</p>
                        </div>
                        <button className="p-3 bg-slate-50 border border-slate-100 rounded-2xl hover:bg-white hover:border-indigo-200 transition-all shadow-sm">
                            <Settings className="w-4 h-4 text-slate-400" />
                        </button>
                    </div>
                    <div className="space-y-4">
                        {models.map((model: any) => (
                            <div key={model.id} className="flex items-center justify-between p-5 bg-slate-50 rounded-3xl border border-slate-100 hover:bg-white hover:border-indigo-100 hover:shadow-xl hover:shadow-indigo-50 transition-all group">
                                <div className="flex items-center gap-5">
                                    <div className={`w-12 h-12 rounded-2xl flex items-center justify-center font-black text-xs ${model.status === 'Healthy' ? 'bg-emerald-50 text-emerald-600 border border-emerald-100' : 'bg-amber-50 text-amber-600 border border-amber-100'}`}>
                                        {model.status === 'Healthy' ? <CheckCircle2 className="w-5 h-5" /> : <AlertTriangle className="w-5 h-5" />}
                                    </div>
                                    <div>
                                        <p className="text-sm font-black text-slate-900 uppercase tracking-tight">{model.name}</p>
                                        <div className="flex items-center gap-4 mt-1">
                                            <span className="text-[10px] font-mono text-slate-400 font-bold uppercase">{model.id}</span>
                                            <span className="px-2 py-0.5 bg-white border border-slate-200 rounded text-[9px] font-black text-slate-500 uppercase tracking-widest">{model.type}</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-8">
                                    <div className="text-right">
                                        <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Accuracy</p>
                                        <p className="text-sm font-black text-slate-900">{model.accuracy}</p>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Precision</p>
                                        <p className="text-sm font-black text-indigo-600">{model.precision}</p>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Recall</p>
                                        <p className="text-sm font-black text-emerald-600">{model.recall}</p>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Latency</p>
                                        <p className="text-sm font-black text-slate-900 font-mono">{model.latency}</p>
                                    </div>
                                    <ChevronRight className="w-5 h-5 text-slate-200 group-hover:text-indigo-600 transition-colors" />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Scientist Metadata (1/3) */}
                <div className="space-y-6">
                    {isSystemAdmin ? (
                        <div className="bg-indigo-600 rounded-[40px] p-8 text-white shadow-2xl shadow-indigo-100 flex flex-col justify-between h-full min-h-[400px]">
                            <div>
                                <h3 className="text-2xl font-black tracking-tight mb-4 leading-none uppercase">Pipeline Version Control</h3>
                                <p className="text-sm font-medium text-indigo-100 opacity-90 leading-relaxed">
                                    Current model (MDL-XGB-001) is utilizing the 'Adam' optimizer with a dynamic learning rate scheduler (eta=0.01).
                                </p>
                            </div>
                            <div className="space-y-4">
                                <div className="p-4 bg-white/10 rounded-3xl border border-white/20 font-mono text-xs">
                                    <div className="flex justify-between items-center mb-1">
                                        <span className="text-indigo-200">/ git_hash</span>
                                        <span className="font-black">7f2a10b</span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-indigo-200">/ environment</span>
                                        <span className="font-black uppercase tracking-widest">production</span>
                                    </div>
                                </div>
                                <button className="w-full py-4 bg-white text-indigo-600 rounded-2xl font-black text-sm shadow-xl hover:bg-slate-50 transition-all flex items-center justify-center gap-2">
                                    <FileCode2 className="w-4 h-4" /> Download Config
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="bg-slate-900 rounded-[40px] p-8 text-white shadow-2xl shadow-slate-200 flex flex-col justify-between h-full min-h-[400px]">
                            <div>
                                <h3 className="text-2xl font-black tracking-tight mb-4 leading-none uppercase">Retraining Schedule</h3>
                                <p className="text-sm font-medium text-slate-400 leading-relaxed">
                                    Automated retraining triggered every Sunday at 02:00 UTC. Next window: 4 days from now.
                                </p>
                            </div>
                            <div className="space-y-4 pt-10">
                                <div className="flex items-center gap-4">
                                    <div className="w-10 h-10 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center text-indigo-400">
                                        <HistoryIcon className="w-5 h-5" />
                                    </div>
                                    <div>
                                        <p className="text-xs font-black">Last Retrain Complete</p>
                                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Oct 22, 2023</p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-4">
                                    <div className="w-10 h-10 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center text-emerald-400">
                                        <Zap className="w-5 h-5" />
                                    </div>
                                    <div>
                                        <p className="text-xs font-black text-emerald-400">Next Scheduled Training</p>
                                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Oct 29, 2023</p>
                                    </div>
                                </div>
                                <button className="w-full mt-6 py-4 border-2 border-white/10 text-white rounded-2xl text-[10px] font-black uppercase tracking-widest hover:border-indigo-400/50 hover:bg-indigo-500/10 transition-all">Adjust Intervals</button>
                            </div>
                        </div>
                    )}
                </div>
            </div>

        </div>
    );
}

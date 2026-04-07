'use client';

import {
    Network,
    Search,
    Bell,
    ChevronDown,
    Maximize2,
    Minus,
    Plus,
    RefreshCw,
    X,
    CreditCard,
    User,
    Smartphone,
    ExternalLink,
    Zap,
    Navigation,
    ChevronRight,
    AlertTriangle,
    ShieldAlert,
    Clock
} from 'lucide-react';
import { useState, useMemo } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { toast } from 'sonner';

const initialNodes = [
    { id: 'CLUSTER-A-392', type: 'cluster', icon: Network, color: 'red', name: 'Cluster #A-392', risk: 'High', score: 94, city: 'Kumasi', top: '50%', left: '50%' },
    { id: 'DEV-9921', type: 'device', icon: Smartphone, color: 'purple', name: 'iPhone 14 Pro', risk: 'Medium', score: 45, city: 'Accra', top: '35%', left: '43%' },
    { id: 'ACC-8821', type: 'account', icon: User, color: 'blue', name: 'Kwame Mensah', risk: 'Low', score: 12, city: 'Legon', top: '40%', left: '62%' },
    { id: 'MERCH-552', type: 'merchant', icon: CreditCard, color: 'emerald', name: 'Jumia GH', risk: 'Low', score: 5, city: 'Adabraka', top: '68%', left: '58%' },
    { id: 'ACC-1102', type: 'account', icon: User, color: 'amber', name: 'Hidden Account', risk: 'Medium', score: 62, city: 'Osu', top: '65%', left: '50%' },
];

export default function NetworkExplorer() {
    const [selectedId, setSelectedId] = useState<string | null>('CLUSTER-A-392');
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);

    const selectedNode = useMemo(() =>
        initialNodes.find(n => n.id === selectedId) || initialNodes[0],
        [selectedId]);

    const handleNodeClick = (id: string) => {
        setSelectedId(id);
        setIsSidebarOpen(true);
        toast.info(`Inspecting ${id}`);
    };

    return (
        <div className="h-screen w-full flex flex-col bg-slate-50 overflow-hidden">

            {/* Top Navigation */}
            <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 shrink-0 z-20 relative shadow-sm">
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
                            <Network className="w-5 h-5" />
                        </div>
                        <div>
                            <h1 className="font-bold text-slate-900 leading-tight">Fraud Intelligence</h1>
                            <p className="text-xs text-slate-500 font-medium">Command Center</p>
                        </div>
                    </div>
                    <div className="w-px h-8 bg-slate-200" />
                    <button className="flex items-center gap-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 px-3 py-1.5 rounded-lg border border-transparent hover:border-slate-200 transition-all">
                        Region: 🇬🇭 Ghana <ChevronDown className="w-4 h-4 text-slate-400" />
                    </button>
                </div>

                <div className="flex-1 max-w-xl px-8">
                    <div className="relative">
                        <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                        <input
                            type="text"
                            placeholder="Search Entity ID, Transaction Hash, or Cluster..."
                            className="w-full pl-9 pr-4 py-2 bg-slate-100 border-none rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none placeholder-slate-400 font-medium"
                        />
                    </div>
                </div>

                <nav className="flex items-center gap-6 text-sm font-semibold text-slate-500">
                    <Link href="/dashboard" className="hover:text-slate-900 transition-colors">Dashboard</Link>
                    <Link href="/network" className="text-blue-600 border-b-2 border-blue-600 pb-5 pt-5">Network Explorer</Link>
                    <Link href="/dashboard/cases" className="hover:text-slate-900 transition-colors">Cases</Link>
                    <div className="w-px h-6 bg-slate-200 ml-2" />
                    <button
                        onClick={() => toast.info('System updates available')}
                        className="text-slate-500 hover:text-slate-900 relative"
                    >
                        <Bell className="w-5 h-5" />
                        <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full border border-white" />
                    </button>
                    <div className="w-8 h-8 rounded-full bg-orange-100 flex items-center justify-center border border-orange-200 overflow-hidden ml-2">
                        <Image src="https://api.dicebear.com/9.x/avataaars/svg?seed=Analyst&backgroundColor=ffdfbf" alt="avatar" width={32} height={32} className="w-full h-full object-cover" />
                    </div>
                </nav>
            </header>

            {/* Main Layout Area */}
            <div className="flex-1 flex overflow-hidden">

                {/* Left Sidebar - Controls */}
                <aside className="w-72 bg-white border-r border-slate-200 flex flex-col shrink-0 z-10 overflow-y-auto">
                    <div className="p-5 border-b border-slate-100">
                        <h2 className="font-bold text-slate-900 text-lg uppercase tracking-tighter">Graph Protocol</h2>
                        <p className="text-[10px] text-slate-400 font-black uppercase tracking-widest mt-1">Visualization parameters</p>
                    </div>

                    <div className="p-5 space-y-8">
                        {/* Analysis Mode */}
                        <div>
                            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3">Analysis Mode</h3>
                            <div className="space-y-1">
                                <button
                                    onClick={() => toast('Cluster Analysis mode enabled')}
                                    className="w-full flex items-center gap-3 px-3 py-2.5 bg-blue-50 text-blue-700 rounded-lg border border-blue-100 font-bold text-xs uppercase tracking-widest transition-all hover:shadow-md"
                                >
                                    <Network className="w-4 h-4" /> Cluster Logic
                                </button>
                                <button
                                    onClick={() => toast('Temporal Flow simulation starting')}
                                    className="w-full flex items-center gap-3 px-3 py-2.5 text-slate-600 hover:bg-slate-50 rounded-lg font-bold text-xs uppercase tracking-widest transition-colors"
                                >
                                    <Zap className="w-4 h-4" /> Temporal Flow
                                </button>
                            </div>
                        </div>

                        {/* Filters */}
                        <div>
                            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3">Network Filters</h3>

                            <div className="bg-slate-50 border border-slate-100 rounded-xl p-4 mb-4">
                                <div className="flex justify-between text-[11px] font-black uppercase tracking-widest text-slate-700 mb-2">
                                    <span>Signal Depth</span>
                                    <span className="text-blue-600">L2</span>
                                </div>
                                <div className="h-1.5 w-full bg-slate-200 rounded-full mt-2 relative overflow-hidden">
                                    <div className="h-full bg-blue-600 rounded-full w-1/3"></div>
                                </div>
                            </div>

                            <div className="space-y-3">
                                <label className="flex items-center gap-3 p-1 cursor-pointer group">
                                    <div className="relative">
                                        <input type="checkbox" className="sr-only" defaultChecked />
                                        <div className="block bg-blue-600 w-10 h-6 rounded-full group-hover:shadow-lg transition-all"></div>
                                        <div className="dot absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition transform translate-x-4"></div>
                                    </div>
                                    <span className="text-[11px] font-black uppercase tracking-widest text-slate-600 group-hover:text-slate-900 transition-colors">Show Txns</span>
                                </label>
                                <label className="flex items-center gap-3 p-1 cursor-pointer group">
                                    <div className="relative">
                                        <input type="checkbox" className="sr-only" defaultChecked />
                                        <div className="block bg-red-600 w-10 h-6 rounded-full group-hover:shadow-lg transition-all"></div>
                                        <div className="dot absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition transform translate-x-4"></div>
                                    </div>
                                    <span className="text-[11px] font-black uppercase tracking-widest text-slate-600 group-hover:text-slate-900 transition-colors">Risk Propagation</span>
                                </label>
                            </div>
                        </div>

                        {/* Legend */}
                        <div>
                            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3">Entity Key</h3>
                            <div className="bg-white border border-slate-100 rounded-xl p-4 space-y-4 shadow-sm">
                                <div className="flex items-center gap-3 text-[10px] font-black uppercase tracking-widest text-slate-600">
                                    <div className="w-2.5 h-2.5 rounded-full bg-blue-600 shadow-sm shadow-blue-200"></div> User Account
                                </div>
                                <div className="flex items-center gap-3 text-[10px] font-black uppercase tracking-widest text-slate-600">
                                    <div className="w-2.5 h-2.5 rounded-full bg-purple-500 shadow-sm shadow-purple-200"></div> Device DNA
                                </div>
                                <div className="flex items-center gap-3 text-[10px] font-black uppercase tracking-widest text-slate-600">
                                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-sm shadow-emerald-200"></div> Merchant Hub
                                </div>
                                <div className="flex items-center gap-3 text-[10px] font-black uppercase tracking-widest text-slate-600">
                                    <div className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-sm shadow-red-200 animate-pulse"></div> High Target
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="mt-auto p-5 border-t border-slate-100 bg-slate-50/50">
                        <button
                            onClick={() => toast.success('Graph view reset to default')}
                            className="w-full flex justify-center items-center gap-2 py-3 px-4 bg-white border border-slate-200 rounded-xl shadow-sm text-[11px] font-black uppercase tracking-widest text-slate-600 hover:bg-slate-50 transition-all active:scale-95"
                        >
                            <RefreshCw className="w-4 h-4" /> Hard Reset
                        </button>
                    </div>
                </aside>

                {/* Center - Graph Area */}
                <div className="flex-1 relative bg-slate-50/50">

                    {/* Graph Toolbar Overlay */}
                    <div className="absolute top-6 left-6 right-6 flex justify-between z-10 pointer-events-none">
                        <div className="bg-white/80 backdrop-blur rounded-2xl shadow-xl border border-slate-200 px-5 py-2.5 pointer-events-auto flex items-center text-[10px] font-black uppercase tracking-widest">
                            <span className="text-slate-400 mr-2">Signal:</span>
                            <span className="text-blue-600 mr-2">{selectedNode.name}</span>
                            <ChevronRight className="w-3 h-3 text-slate-300 mx-1" />
                            <span className="text-slate-900">{selectedNode.id}</span>
                        </div>

                        <div className="flex gap-2 pointer-events-auto">
                            <button
                                onClick={() => toast.success('Full intelligence snapshot captured')}
                                className="bg-white border border-slate-200 px-5 py-2.5 rounded-2xl text-[10px] font-black uppercase tracking-widest text-slate-700 shadow-lg hover:bg-slate-50 transition-all active:scale-95"
                            >
                                Snapshot
                            </button>
                            <button
                                onClick={() => toast.success('Cluster frozen for review')}
                                className="bg-slate-900 border border-transparent px-5 py-2.5 rounded-2xl text-[10px] font-black uppercase tracking-widest text-white shadow-xl hover:bg-slate-800 transition-all flex items-center gap-2 active:scale-95"
                            >
                                <ShieldAlert className="w-3.5 h-3.5" /> Isolated View
                            </button>
                        </div>
                    </div>

                    {/* Graph Simulation Canvas */}
                    <div className="absolute inset-0 overflow-hidden flex items-center justify-center">

                        {/* Lines/Edges */}
                        <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 0 }}>
                            {initialNodes.slice(1).map((node, i) => (
                                <line
                                    key={i}
                                    x1="50%" y1="50%"
                                    x2={node.left} y2={node.top}
                                    stroke={node.risk === 'High' ? '#ef4444' : '#cbd5e1'}
                                    strokeWidth={node.risk === 'High' ? '2' : '1'}
                                    strokeDasharray={node.risk === 'High' ? '4,4' : ''}
                                    className="transition-all duration-500"
                                />
                            ))}
                        </svg>

                        <div className="relative w-full h-full" style={{ zIndex: 1 }}>
                            {initialNodes.map((node) => {
                                const Icon = node.icon;
                                const isSelected = selectedId === node.id;

                                return (
                                    <div
                                        key={node.id}
                                        className="absolute transform -translate-x-1/2 -translate-y-1/2 z-20 transition-all duration-500 hover:scale-110"
                                        style={{ top: node.top, left: node.left }}
                                        onClick={() => handleNodeClick(node.id)}
                                    >
                                        {node.type === 'cluster' && (
                                            <div className="w-32 h-32 bg-red-500/10 rounded-full absolute -top-8 -left-8 animate-pulse"></div>
                                        )}
                                        <div className={`relative ${node.type === 'cluster' ? 'w-16 h-16' : 'w-10 h-10'} rounded-full flex items-center justify-center cursor-pointer shadow-xl transition-all border-2
                                            ${isSelected ? 'scale-125 ring-4 ring-blue-500/20' : ''}
                                            ${node.color === 'red' ? 'bg-red-50 border-red-500 text-red-600' :
                                                node.color === 'purple' ? 'bg-purple-50 border-purple-500 text-purple-600' :
                                                    node.color === 'blue' ? 'bg-blue-50 border-blue-500 text-blue-600' :
                                                        node.color === 'emerald' ? 'bg-emerald-50 border-emerald-500 text-emerald-600' :
                                                            'bg-amber-50 border-amber-500 text-amber-600'}`}>
                                            <Icon className={node.type === 'cluster' ? 'w-8 h-8' : 'w-5 h-5'} />
                                        </div>
                                        <div className={`absolute top-full left-1/2 transform -translate-x-1/2 mt-2 bg-white border border-slate-200 shadow-xl rounded-lg px-2.5 py-1 text-[9px] font-black uppercase tracking-widest text-slate-700 whitespace-nowrap transition-all ${isSelected ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-1 pointer-events-none group-hover:opacity-100'}`}>
                                            {node.name}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Map Controls */}
                    <div className="absolute bottom-8 left-8 flex flex-col gap-2 z-10">
                        <div className="bg-white/80 backdrop-blur border border-slate-200 rounded-2xl shadow-2xl overflow-hidden flex flex-col">
                            <button onClick={() => toast('Zoomed In')} className="p-3 hover:bg-white text-slate-600 border-b border-slate-100 transition-colors">
                                <Plus className="w-5 h-5" />
                            </button>
                            <button onClick={() => toast('Zoomed Out')} className="p-3 hover:bg-white text-slate-600 border-b border-slate-100 transition-colors">
                                <Minus className="w-5 h-5" />
                            </button>
                            <button onClick={() => toast('Fit to screen')} className="p-3 hover:bg-white text-slate-600 transition-colors">
                                <Maximize2 className="w-4 h-4 m-0.5" />
                            </button>
                        </div>
                    </div>

                </div>

                {/* Right Sidebar - Details Panel */}
                {isSidebarOpen && (
                    <aside className="w-80 bg-white border-l border-slate-200 flex flex-col shrink-0 z-10 shadow-[-10px_0_30px_rgba(0,0,0,0.02)] relative animate-in slide-in-from-right duration-300">

                        <div className={`p-6 border-b border-slate-100 flex items-start justify-between absolute top-0 w-full z-10 backdrop-blur-md
                            ${selectedNode.risk === 'High' ? 'bg-red-50/80 text-red-900 border-red-100' : 'bg-slate-50/80 text-slate-900 border-slate-200'}`}>
                            <div>
                                <h2 className="font-black flex items-center gap-2 uppercase tracking-tighter text-lg">
                                    <ShieldAlert className={`w-5 h-5 ${selectedNode.risk === 'High' ? 'text-red-600' : 'text-blue-600'}`} />
                                    Entity DNA
                                </h2>
                                <p className="text-[10px] mt-1 font-black uppercase tracking-widest text-slate-400">UUID: {selectedNode.id}</p>
                            </div>
                            <button
                                onClick={() => setIsSidebarOpen(false)}
                                className="text-slate-400 hover:text-slate-900 transition-colors p-1"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="flex-1 overflow-y-auto pt-[100px] pb-6 px-6 space-y-8 scrollbar-hide">

                            {/* Probability Score */}
                            <div>
                                <div className="flex justify-between items-end mb-2">
                                    <span className="text-[11px] font-black uppercase tracking-widest text-slate-500">Cognitive Risk Index</span>
                                    <span className={`text-4xl font-black leading-none ${selectedNode.risk === 'High' ? 'text-red-600' : 'text-blue-600'}`}>
                                        {selectedNode.score}%
                                    </span>
                                </div>
                                <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden mb-3">
                                    <div className={`h-full rounded-full transition-all duration-1000 shadow-lg ${selectedNode.risk === 'High' ? 'bg-red-600 shadow-red-200' : 'bg-blue-600 shadow-blue-200'}`} style={{ width: `${selectedNode.score}%` }}></div>
                                </div>
                                <p className={`text-[10px] font-black uppercase tracking-widest flex items-center gap-1.5 ${selectedNode.risk === 'High' ? 'text-red-600' : 'text-emerald-600'}`}>
                                    {selectedNode.risk === 'High' ? <Zap className="w-3 h-3 animate-bounce" /> : <ShieldAlert className="w-3 h-3" />}
                                    {selectedNode.risk === 'High' ? 'Critical Velocity Detected' : 'Signal Stabilized'}
                                </p>
                            </div>

                            {/* Geography */}
                            <div>
                                <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3">Geospatial Origin</h3>
                                <div className="bg-slate-50 rounded-2xl p-4 border border-slate-100 flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-lg bg-white border border-slate-200 flex items-center justify-center shadow-sm">
                                            <Navigation className="w-4 h-4 text-blue-500" />
                                        </div>
                                        <div className="flex flex-col">
                                            <span className="text-xs font-black text-slate-900 uppercase">Ghana</span>
                                            <span className="text-[10px] font-bold text-slate-500 uppercase">{selectedNode.city} Region</span>
                                        </div>
                                    </div>
                                    <span className="text-[10px] font-black uppercase text-blue-600 underline cursor-pointer">View Map</span>
                                </div>
                            </div>

                            {/* Recent Intelligence */}
                            <div>
                                <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3">Recent Intelligence Signals</h3>
                                <div className="space-y-4">
                                    <div className="flex gap-4 relative">
                                        <div className="absolute left-[9px] top-6 w-px h-8 bg-slate-100"></div>
                                        <div className="shrink-0 w-5 h-5 rounded-full bg-red-100 flex items-center justify-center">
                                            <AlertTriangle className="w-3 h-3 text-red-600" />
                                        </div>
                                        <div>
                                            <h4 className="text-[11px] font-black uppercase text-slate-900 mt-0.5 tracking-tight">Rapid structuring detected</h4>
                                            <p className="text-[10px] text-slate-400 mt-0.5 font-medium leading-relaxed">14 cross-border txns in 8 mins</p>
                                        </div>
                                    </div>
                                    <div className="flex gap-4">
                                        <div className="shrink-0 w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center">
                                            <Clock className="w-3 h-3 text-blue-600" />
                                        </div>
                                        <div>
                                            <h4 className="text-[11px] font-black uppercase text-slate-900 mt-0.5 tracking-tight">Temporal Outlier</h4>
                                            <p className="text-[10px] text-slate-400 mt-0.5 font-medium leading-relaxed">System access at 03:22 AM GMT</p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                        </div>

                        <div className="p-6 border-t border-slate-200 bg-slate-50/50">
                            <Link href="/dashboard/cases" className="w-full flex justify-center items-center gap-2 py-3.5 px-4 rounded-2xl shadow-xl text-xs font-black uppercase tracking-widest text-white bg-slate-900 hover:bg-black transition-all active:scale-95">
                                Initialize Case <ExternalLink className="w-3.5 h-3.5" />
                            </Link>
                        </div>

                    </aside>
                )}

            </div>
        </div>
    );
}


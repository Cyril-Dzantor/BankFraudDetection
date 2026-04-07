'use client';

import {
    FolderOpen,
    Plus,
    Search,
    Filter,
    Clock,
    User,
    AlertTriangle,
    FileText,
    CheckCircle2,
    Calendar,
    Building2,
    Briefcase,
    Tag,
    ArrowRight,
    Network,
    ExternalLink
} from 'lucide-react';
import { useState, useMemo, useEffect } from 'react';
import { toast } from 'sonner';
import { useRole } from '@/app/context/RoleContext';
import { getBackendUrl, getOrchestratorUrl } from '@/app/utils/api';
import Link from 'next/link';
import Modal from '../../components/Modal';

export default function CaseManagementPage() {
    const { role } = useRole();
    const [cases, setCases] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isNewCaseModalOpen, setIsNewCaseModalOpen] = useState(false);
    const [isCaseDetailModalOpen, setIsCaseDetailModalOpen] = useState(false);
    const [selectedCase, setSelectedCase] = useState<any>(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [activeTab, setActiveTab] = useState<'details' | 'graph'>('details');
    type NetworkNode = {
        id: string;
        name: string;
        type: 'account' | 'device' | 'merchant' | 'cluster' | string;
        score: number;
        risk: 'Low' | 'Medium' | 'High' | 'Critical' | string;
        city?: string;
        top: string;
        left: string;
        color?: string;
    };

    type NetworkLink = {
        source: string;
        target: string;
        risk: 'Low' | 'Medium' | 'High' | 'Critical' | string;
    };

    const [networkData, setNetworkData] = useState<{ nodes: NetworkNode[]; links: NetworkLink[] } | null>(null);
    const [isLoadingGraph, setIsLoadingGraph] = useState(false);
    const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);

    // Form State
    const [newCase, setNewCase] = useState({
        title: '',
        type: 'Retail',
        priority: 'Medium',
        value: ''
    });

    const backendUrl = getBackendUrl();
    const orchestratorUrl = getOrchestratorUrl();

    useEffect(() => {
        const fetchCases = async (isPolling = false) => {
            if (!isPolling) setIsLoading(true);
            try {
                const res = await fetch(`${backendUrl}/api/v1/cases/`);
                const data = await res.json();
                if (data.items) {
                    setCases(data.items);
                }
            } catch (error) {
                console.error('Error fetching cases:', error);
                if (!isPolling) toast.error('Could not load cases from backend.');
            } finally {
                if (!isPolling) setIsLoading(false);
            }
        };
        fetchCases();

        const intervalId = setInterval(() => fetchCases(true), 3000);
        return () => clearInterval(intervalId);
    }, [backendUrl]);

    const filteredCases = useMemo(() => {
        return cases.filter(c => {
            const searchLower = searchTerm.toLowerCase();
            return c.title.toLowerCase().includes(searchLower) ||
                c.id.toLowerCase().includes(searchLower) ||
                c.status.toLowerCase().includes(searchLower) ||
                c.priority.toLowerCase().includes(searchLower) ||
                c.assignee.toLowerCase().includes(searchLower) ||
                c.type.toLowerCase().includes(searchLower);
        });
    }, [cases, searchTerm]);

    const handleCreateCase = (e: React.FormEvent) => {
        e.preventDefault();
        if (!newCase.title) {
            toast.error("Please provide a case subject");
            return;
        }
        const id = `CAS-2023-${Math.floor(Math.random() * 9000) + 1000}`;
        const createdCase = {
            id,
            title: newCase.title,
            assignee: 'Me',
            status: 'New',
            priority: newCase.priority,
            created: 'Just now',
            updated: 'Just now',
            value: `GHS ${newCase.value || '0'}`,
            type: newCase.type,
            tags: 'New, Pending Correlation'
        };
        setCases([createdCase, ...cases]);
        setIsNewCaseModalOpen(false);
        setNewCase({ title: '', type: 'Retail', priority: 'Medium', value: '' });
        toast.success(`Case ${id} created`);
    };

    const loadGraphData = async () => {
        setActiveTab('graph');
        if (networkData || !selectedCase) return;
        setIsLoadingGraph(true);
        try {
            const res = await fetch(`${backendUrl}/api/v1/network/topology/${selectedCase.id}`);
            const data = await res.json();
            setNetworkData(data);
        } catch (error) {
            console.error(error);
            toast.error('Failed to load network topology');
        } finally {
            setIsLoadingGraph(false);
        }
    };

    const handleCaseUpdate = async (field: string, value: string) => {
        if (!selectedCase) return;
        try {
            const res = await fetch(`${backendUrl}/api/v1/cases/${selectedCase.id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ [field]: value })
            });
            if (res.ok) {
                const updatedCase = await res.json();
                setSelectedCase(updatedCase);
                setCases(cases.map(c => c.id === updatedCase.id ? updatedCase : c));
                toast.success(`Case ${field} updated`);
            }
        } catch (error) {
            console.error(error);
            toast.error('Failed to update case');
        }
    };

    const handleResolution = async (resolution: string) => {
        if (!selectedCase) return;
        try {
            await handleCaseUpdate('status', 'Resolved');
            await fetch(`${backendUrl}/api/v1/feedback/decisions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    transaction_id: selectedCase.id,
                    alert_id: selectedCase.id,
                    original_decision: 'CHALLENGE',
                    final_bank_outcome: resolution,
                    analyst_label: resolution,
                    notes: selectedCase.notes || ''
                })
            });
            toast.success(`Models updated with ${resolution}`);
            setIsCaseDetailModalOpen(false);
        } catch (error) {
            console.error(error);
            toast.error('Failed to send feedback');
        }
    };

    return (
        <div className="max-w-7xl mx-auto space-y-6 pb-12">
            {/* Header Bar */}
            <div className="flex items-start justify-between">
                <div>
                    <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Case Management</h1>
                    <p className="text-slate-500 text-sm mt-1">Investigate, track, and resolve structural fraud cases.</p>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => toast.success('Report generating...')}
                        className="flex items-center gap-2 px-4 py-2 border border-slate-200 bg-white rounded-lg text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50 transition-colors"
                    >
                        <FileText className="w-4 h-4" /> Export
                    </button>
                    {(role === 'senior_analyst' || role === 'system_admin') && (
                        <button
                            onClick={() => setIsNewCaseModalOpen(true)}
                            className="flex items-center gap-2 px-4 py-2 bg-blue-600 border border-transparent rounded-lg text-sm font-semibold text-white shadow-sm hover:bg-blue-700 transition-all active:scale-95"
                        >
                            <Plus className="w-4 h-4" /> New Case
                        </button>
                    )}
                </div>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm font-bold">
                    <span className="text-slate-500 text-xs uppercase tracking-widest block mb-2">Active Cases</span>
                    <h3 className="text-3xl text-slate-900">{isLoading ? '...' : cases.filter(c => c.status !== 'Resolved').length}</h3>
                </div>
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm font-bold">
                    <span className="text-slate-500 text-xs uppercase tracking-widest block mb-2">Pending</span>
                    <h3 className="text-3xl text-slate-900">{cases.filter(c => c.status === 'New').length}</h3>
                </div>
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm font-bold">
                    <span className="text-slate-500 text-xs uppercase tracking-widest block mb-2">Resolved</span>
                    <h3 className="text-3xl text-slate-900">{cases.filter(c => c.status === 'Resolved').length}</h3>
                </div>
            </div>

            {/* List */}
            <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden min-h-[400px]">
                <table className="w-full text-sm text-left">
                    <thead className="bg-slate-50 border-b border-slate-200 text-xs text-slate-500 font-bold uppercase tracking-wider">
                        <tr>
                            <th className="px-6 py-4">Case Identity</th>
                            <th className="px-6 py-4">Status</th>
                            <th className="px-6 py-4">Assignee</th>
                            <th className="px-6 py-4 text-right">Action</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                        {filteredCases.map(c => (
                            <tr key={c.id} onClick={() => { setSelectedCase(c); setIsCaseDetailModalOpen(true); }} className="hover:bg-slate-50 cursor-pointer transition-colors group">
                                <td className="px-6 py-4">
                                    <span className="font-mono text-xs font-bold text-blue-600 block mb-1">{c.id}</span>
                                    <span className="font-bold text-slate-900">{c.title}</span>
                                </td>
                                <td className="px-6 py-4">
                                    <span className={`px-2 py-1 rounded-lg text-[10px] font-black uppercase tracking-widest border
                                        ${c.status === 'Resolved' ? 'bg-emerald-50 text-emerald-700 border-emerald-100' : 'bg-blue-50 text-blue-700 border-blue-100'}`}>
                                        {c.status}
                                    </span>
                                </td>
                                <td className="px-6 py-4 text-slate-600 font-bold">{c.assignee}</td>
                                <td className="px-6 py-4 text-right">
                                    <ArrowRight className="w-4 h-4 ml-auto text-slate-300 group-hover:text-slate-600" />
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Details Modal */}
            <Modal
                isOpen={isCaseDetailModalOpen}
                onClose={() => { setIsCaseDetailModalOpen(false); setActiveTab('details'); }}
                title={selectedCase ? `Forensic Case: ${selectedCase.id}` : "Case"}
                footer={(
                    <div className="flex justify-between w-full">
                        <button onClick={() => setIsCaseDetailModalOpen(false)} className="px-4 py-2 border border-slate-200 rounded-lg text-sm font-bold hover:bg-slate-50">Close</button>
                        {selectedCase?.status !== 'Resolved' && role !== 'junior_analyst' && (
                            <div className="flex gap-2">
                                <button onClick={() => handleResolution('Safe')} className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-sm font-black uppercase hover:bg-emerald-700">Dismiss</button>
                                <button onClick={() => handleResolution('Fraud')} className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-black uppercase hover:bg-red-700">Confirm Fraud</button>
                            </div>
                        )}
                    </div>
                )}
            >
                {selectedCase && (
                    <div className="space-y-6">
                        <div className="flex border-b border-slate-200 gap-4 mb-4">
                            <button onClick={() => setActiveTab('details')} className={`pb-2 text-xs font-black uppercase tracking-widest border-b-2 transition-colors ${activeTab === 'details' ? 'border-blue-600 text-blue-600' : 'border-transparent text-slate-400'}`}>Summary</button>
                            <button onClick={loadGraphData} className={`pb-2 text-xs font-black uppercase tracking-widest border-b-2 transition-colors ${activeTab === 'graph' ? 'border-blue-600 text-blue-600' : 'border-transparent text-slate-400'}`}>Graph Analysis</button>
                        </div>

                        {activeTab === 'details' ? (
                            <div className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="text-[10px] font-black text-slate-400 uppercase mb-1 block">Assignee</label>
                                        <div className="p-3 bg-slate-50 rounded-xl font-bold text-sm border border-slate-200">{selectedCase.assignee}</div>
                                    </div>
                                    <div>
                                        <label className="text-[10px] font-black text-slate-400 uppercase mb-1 block">Value</label>
                                        <div className="p-3 bg-slate-50 rounded-xl font-mono text-sm border border-slate-200">{selectedCase.value}</div>
                                    </div>
                                </div>
                                <div>
                                    <label className="text-[10px] font-black text-slate-400 uppercase mb-1 block">Investigation Notes</label>
                                    <textarea
                                        className="w-full h-32 p-4 bg-slate-50 border border-slate-200 rounded-xl text-sm outline-none focus:ring-2 focus:ring-blue-500"
                                        placeholder="Forensic trail..."
                                        value={selectedCase.notes || ''}
                                        onChange={(e) => setSelectedCase({ ...selectedCase, notes: e.target.value })}
                                        onBlur={(e) => handleCaseUpdate('notes', e.target.value)}
                                    />
                                </div>
                                <div className="pt-4 border-t border-slate-100 space-y-3">
                                    <Link
                                        href={`/dashboard/accounts/${selectedCase.customer_id || selectedCase.id}`}
                                        className="flex items-center justify-center gap-2 w-full py-3 bg-slate-900 text-white rounded-xl text-xs font-black uppercase tracking-widest hover:bg-slate-800 transition-all font-bold"
                                    >
                                        Full Forensic Profile <ExternalLink className="w-4 h-4" />
                                    </Link>
                                    <p className="text-[10px] text-slate-400 text-center uppercase font-bold tracking-widest opacity-60 italic">Cross-reference with behavioral telemetry</p>
                                </div>
                            </div>
                        ) : (
                            <div className="bg-slate-900 rounded-2xl h-[420px] flex flex-col md:flex-row gap-4 p-4 relative overflow-hidden border border-slate-800">
                                {isLoadingGraph && (
                                    <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 z-20">
                                        <span className="text-white text-xs font-black animate-pulse uppercase tracking-[0.2em]">
                                            Querying Graph...
                                        </span>
                                    </div>
                                )}

                                {!isLoadingGraph && !networkData && (
                                    <div className="flex-1 flex items-center justify-center">
                                        <span className="text-slate-500 text-xs font-black uppercase">
                                            No telemetry or graph context available for this case
                                        </span>
                                    </div>
                                )}

                                {networkData && !isLoadingGraph && (
                                    <>
                                        {/* Graph canvas */}
                                        <div className="relative flex-1 min-h-[260px] rounded-xl bg-slate-950/40 border border-slate-800/80 overflow-hidden">
                                            <svg
                                                viewBox="0 0 100 100"
                                                className="absolute inset-0 w-full h-full"
                                            >
                                                {/* Edges */}
                                                {networkData.links.map((link, idx) => {
                                                    const source = networkData.nodes.find(n => n.id === link.source);
                                                    const target = networkData.nodes.find(n => n.id === link.target);
                                                    if (!source || !target) return null;

                                                    const sx = parseFloat(source.left);
                                                    const sy = parseFloat(source.top);
                                                    const tx = parseFloat(target.left);
                                                    const ty = parseFloat(target.top);

                                                    const strokeColor =
                                                        link.risk === 'Critical'
                                                            ? '#ef4444'
                                                            : link.risk === 'High'
                                                            ? '#fb923c'
                                                            : link.risk === 'Medium'
                                                            ? '#eab308'
                                                            : '#22c55e';

                                                    return (
                                                        <g key={idx}>
                                                            <line
                                                                x1={sx}
                                                                y1={sy}
                                                                x2={tx}
                                                                y2={ty}
                                                                stroke={strokeColor}
                                                                strokeWidth={0.8}
                                                                strokeOpacity={0.6}
                                                            />
                                                        </g>
                                                    );
                                                })}

                                                {/* Nodes */}
                                                {networkData.nodes.map(node => {
                                                    const x = parseFloat(node.left);
                                                    const y = parseFloat(node.top);

                                                    const isCenter = node.id.startsWith('ACT-') && node.id.includes(selectedCase.id);

                                                    const fillColor =
                                                        node.type === 'device' || node.type === 'cluster'
                                                            ? '#f97316'
                                                            : node.type === 'merchant'
                                                            ? '#22c55e'
                                                            : node.risk === 'Critical'
                                                            ? '#ef4444'
                                                            : node.risk === 'High'
                                                            ? '#f97316'
                                                            : node.risk === 'Medium'
                                                            ? '#eab308'
                                                            : '#38bdf8';

                                                    const radius = isCenter ? 4.2 : Math.max(2.4, Math.min(4, node.score / 30));

                                                    return (
                                                        <g
                                                            key={node.id}
                                                            className="cursor-pointer"
                                                            onClick={() => setSelectedNode(node)}
                                                        >
                                                            <circle
                                                                cx={x}
                                                                cy={y}
                                                                r={radius + 0.6}
                                                                fill="#020617"
                                                                stroke={fillColor}
                                                                strokeWidth={0.5}
                                                                opacity={selectedNode && selectedNode.id !== node.id ? 0.4 : 0.9}
                                                            />
                                                            <circle
                                                                cx={x}
                                                                cy={y}
                                                                r={radius}
                                                                fill={fillColor}
                                                                opacity={selectedNode && selectedNode.id !== node.id ? 0.4 : 0.95}
                                                            />
                                                        </g>
                                                    );
                                                })}
                                            </svg>

                                            {/* Inline labels for key nodes */}
                                            {networkData.nodes.map(node => (
                                                <div
                                                    key={node.id}
                                                    style={{
                                                        position: 'absolute',
                                                        top: node.top,
                                                        left: node.left,
                                                        transform: 'translate(-50%, -140%)'
                                                    }}
                                                    className="pointer-events-none"
                                                >
                                                    <span className="px-1.5 py-0.5 rounded-full bg-slate-900/70 border border-slate-700 text-[9px] font-semibold text-slate-100 shadow-sm whitespace-nowrap">
                                                        {node.name}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>

                                        {/* Analyst helper panel */}
                                        <div className="w-full md:w-72 flex flex-col gap-3 text-xs text-slate-100">
                                            <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 space-y-2">
                                                <div className="flex items-center justify-between">
                                                    <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">
                                                        Case-Specific Threat Topology
                                                    </span>
                                                    <span className="px-1.5 py-0.5 rounded-full bg-blue-900/50 text-[9px] font-bold uppercase tracking-wide text-blue-200 border border-blue-800/70">
                                                        Synthetic
                                                    </span>
                                                </div>

                                                {(() => {
                                                    const isAto = networkData.nodes.some(n => n.id.startsWith('DEV-ATO'));
                                                    const patternLabel = isAto ? 'Pattern A · Account Takeover Ring' : 'Pattern B · Money Mule Network';
                                                    const patternDesc = isAto
                                                        ? 'Multiple victim accounts share a suspicious device / network endpoint — classic ATO ring signature.'
                                                        : 'Funds appear to bounce through intermediary mules before exiting to an external beneficiary.';

                                                    return (
                                                        <div className="space-y-1.5">
                                                            <p className="text-[11px] font-semibold text-slate-100">
                                                                {patternLabel}
                                                            </p>
                                                            <p className="text-[11px] text-slate-400 leading-relaxed">
                                                                {patternDesc}
                                                            </p>
                                                        </div>
                                                    );
                                                })()}
                                            </div>

                                            <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 space-y-2">
                                                <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">
                                                    Node Insight
                                                </p>
                                                {selectedNode ? (
                                                    <div className="space-y-1.5">
                                                        <p className="text-[11px] font-semibold text-slate-50">
                                                            {selectedNode.name}
                                                        </p>
                                                        <p className="text-[11px] text-slate-400">
                                                            <span className="font-semibold text-slate-300">Type:</span>{' '}
                                                            {selectedNode.type.toUpperCase()} ·{' '}
                                                            <span className="font-semibold text-slate-300">Risk:</span>{' '}
                                                            {selectedNode.risk}
                                                            {selectedNode.city && (
                                                                <span className="ml-1 text-slate-500">· {selectedNode.city}</span>
                                                            )}
                                                        </p>
                                                        <p className="text-[11px] text-slate-400">
                                                            <span className="font-semibold text-slate-300">Connectivity:</span>{' '}
                                                            {networkData.links.filter(
                                                                l => l.source === selectedNode.id || l.target === selectedNode.id
                                                            ).length}{' '}
                                                            linked entities
                                                        </p>
                                                    </div>
                                                ) : (
                                                    <p className="text-[11px] text-slate-500">
                                                        Click any node in the topology to see its role in the fraud pattern.
                                                    </p>
                                                )}
                                            </div>

                                            <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 space-y-2">
                                                <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">
                                                    Legend
                                                </p>
                                                <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-300">
                                                    <div className="flex items-center gap-1.5">
                                                        <span className="w-2.5 h-2.5 rounded-full bg-sky-400" />
                                                        <span>Account (Low / Med)</span>
                                                    </div>
                                                    <div className="flex items-center gap-1.5">
                                                        <span className="w-2.5 h-2.5 rounded-full bg-orange-400" />
                                                        <span>Device / IP</span>
                                                    </div>
                                                    <div className="flex items-center gap-1.5">
                                                        <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
                                                        <span>Merchant / Beneficiary</span>
                                                    </div>
                                                    <div className="flex items-center gap-1.5">
                                                        <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
                                                        <span>Critical Node</span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </Modal>

            {/* New Case Modal Placeholder */}
            <Modal isOpen={isNewCaseModalOpen} onClose={() => setIsNewCaseModalOpen(false)} title="Initialize Case">
                <form onSubmit={handleCreateCase} className="space-y-4">
                    <input type="text" placeholder="Subject" className="w-full p-3 bg-slate-50 rounded-xl border border-slate-200" value={newCase.title} onChange={e => setNewCase({ ...newCase, title: e.target.value })} />
                    <button className="w-full py-3 bg-blue-600 text-white rounded-xl font-black uppercase tracking-widest shadow-lg">Start</button>
                </form>
            </Modal>
        </div>
    );
}

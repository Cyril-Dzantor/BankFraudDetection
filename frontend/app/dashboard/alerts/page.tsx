'use client';

import {
    Bell,
    AlertTriangle,
    AlertCircle,
    Clock,
    ChevronDown,
    Filter,
    Check,
    Download,
    Smartphone,
    CreditCard,
    Globe,
    Landmark,
    Search,
    User,
    ShieldAlert,
    ExternalLink,
    Activity,
    Loader2,
    MapPin
} from 'lucide-react';
import { useState, useMemo, useEffect } from 'react';
import Link from 'next/link';
import { toast } from 'sonner';
import { useRole } from '@/app/context/RoleContext';
import { getBackendUrl } from '@/app/utils/api';
import Modal from '../../components/Modal';

// Helper for icons to avoid serialization issues in SSR
const ChannelIcon = ({ type, className = "w-4 h-4 text-slate-400" }: { type: string, className?: string }) => {
    switch (type) {
        case 'smartphone': return <Smartphone className={className} />;
        case 'globe': return <Globe className={className} />;
        case 'credit-card': return <CreditCard className={className} />;
        case 'landmark': return <Landmark className={className} />;
        default: return <Smartphone className={className} />;
    }
};

interface Alert {
    id: string;
    branch: string;
    customer: string;
    customer_id?: string;
    acctStart: string;
    initials: string;
    amount: string;
    riskLevel: string;
    score: number;
    channel: string;
    channelIconType: string;
    time: string;
    location: string;
    device: string;
    reason: string;
    triggered_rules: string;
    created_at: string;
    transaction_type?: string;
    recipient_account?: string;
    recipient_name?: string;
    transaction_notes?: string;
}

export default function AlertQueueTriage() {
    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
    const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [filterTier, setFilterTier] = useState('All');
    const [approvedCount, setApprovedCount] = useState(0);

    const backendUrl = getBackendUrl();

    // Relative time helper
    const getRelativeTime = (isoString: string) => {
        try {
            const date = new Date(isoString);
            const now = new Date();
            const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

            if (diffInSeconds < 60) return 'just now';
            if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minute${Math.floor(diffInSeconds / 60) > 1 ? 's' : ''} ago`;
            if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hour${Math.floor(diffInSeconds / 3600) > 1 ? 's' : ''} ago`;
            return date.toLocaleDateString();
        } catch (e) {
            return 'just now';
        }
    };

    useEffect(() => {
        const fetchAlerts = async (isPolling = false) => {
            if (!isPolling) setIsLoading(true);
            try {
                const res = await fetch(`${backendUrl}/api/v1/alerts/`);
                const data = await res.json();
                if (data.items) {
                    setAlerts(data.items);
                }
            } catch (error) {
                console.error('Error fetching alerts:', error);
                if (!isPolling) toast.error('Could not load alerts from backend.');
            } finally {
                if (!isPolling) setIsLoading(false);
            }
        };
        fetchAlerts();

        // Auto-refresh every 3 seconds for real-time updates
        const intervalId = setInterval(() => fetchAlerts(true), 3000);
        return () => clearInterval(intervalId);
    }, [backendUrl]);

    const filteredAlerts = useMemo(() => {
        return alerts.filter(alert => {
            const searchLower = searchTerm.toLowerCase();
            const matchesSearch =
                alert.customer.toLowerCase().includes(searchLower) ||
                alert.id.toLowerCase().includes(searchLower) ||
                alert.acctStart.toLowerCase().includes(searchLower) ||
                alert.channel.toLowerCase().includes(searchLower);

            const matchesTier = filterTier === 'All' || alert.riskLevel === filterTier;
            return matchesSearch && matchesTier;
        });
    }, [alerts, searchTerm, filterTier]);

    const handleApprove = (id: string) => {
        setAlerts(prev => prev.filter(a => a.id !== id));
        setApprovedCount(prev => prev + 1);
        toast.success(`Transaction ${id} approved successfully`);
        setIsDetailModalOpen(false);
    };

    const handleReject = (id: string, customer: string) => {
        setAlerts(prev => prev.filter(a => a.id !== id));
        toast.error(`Transaction ${id} for ${customer} flagged as FRAUD`);
        setIsDetailModalOpen(false);
    };

    const handleBulkApprove = () => {
        const count = filteredAlerts.length;
        if (count === 0) return;
        setAlerts(prev => prev.filter(a => !filteredAlerts.find(f => f.id === a.id)));
        setApprovedCount(prev => prev + count);
        toast.success(`${count} alerts bulk-approved successfully`);
    };

    const openDetails = (alert: Alert) => {
        setSelectedAlert(alert);
        setIsDetailModalOpen(true);
    };

    return (
        <div className="max-w-7xl mx-auto space-y-6 pb-12">

            {/* Header Bar */}
            <div className="flex items-start justify-between">
                <div>
                    <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Alert Queue Triage</h1>
                    <p className="text-slate-500 text-sm mt-1">Real-time transaction monitoring for high-risk activities across Ghana regions.</p>
                </div>
                <div className="flex items-center gap-2 bg-emerald-50 border border-emerald-100 px-3 py-1.5 rounded-full shadow-sm">
                    <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="text-xs font-bold text-emerald-700 tracking-wide">System Operational: 99.9% Uptime</span>
                </div>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                    <div className="flex justify-between items-start mb-2">
                        <span className="text-slate-500 text-sm font-medium">Active Alerts</span>
                        <Bell className="w-5 h-5 text-slate-400" />
                    </div>
                    <div className="flex items-center gap-3">
                        <h3 className="text-3xl font-bold text-slate-900 leading-none">
                            {isLoading ? <Loader2 className="w-6 h-6 animate-spin text-slate-300" /> : alerts.length}
                        </h3>
                        <span className="inline-flex items-center gap-1 rounded bg-emerald-50 px-1.5 py-0.5 text-xs font-bold text-emerald-600">
                            Live
                        </span>
                    </div>
                </div>

                <div className="bg-white rounded-xl border-l-4 border-l-red-500 border-y border-r border-slate-200 p-5 shadow-sm">
                    <div className="flex justify-between items-start mb-2">
                        <span className="text-slate-500 text-sm font-medium">Critical Risk</span>
                        <AlertTriangle className="w-5 h-5 text-red-400" />
                    </div>
                    <div className="flex items-center gap-3">
                        <h3 className="text-3xl font-bold text-slate-900 leading-none">
                            {isLoading ? <Loader2 className="w-6 h-6 animate-spin text-slate-300" /> : alerts.filter(a => a.riskLevel === 'Critical').length}
                        </h3>
                    </div>
                </div>

                <div className="bg-white rounded-xl border-l-4 border-l-amber-500 border-y border-r border-slate-200 p-5 shadow-sm">
                    <div className="flex justify-between items-start mb-2">
                        <span className="text-slate-500 text-sm font-medium">High Risk</span>
                        <AlertCircle className="w-5 h-5 text-amber-500" />
                    </div>
                    <div className="flex items-center gap-3">
                        <h3 className="text-3xl font-bold text-slate-900 leading-none">
                            {isLoading ? <Loader2 className="w-6 h-6 animate-spin text-slate-300" /> : alerts.filter(a => a.riskLevel === 'High').length}
                        </h3>
                    </div>
                </div>

                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                    <div className="flex justify-between items-start mb-2">
                        <span className="text-slate-500 text-sm font-medium">Approved / Cleared</span>
                        <Check className="w-5 h-5 text-emerald-500" />
                    </div>
                    <div className="flex items-center gap-3">
                        <h3 className="text-3xl font-bold text-slate-900 leading-none">{approvedCount}</h3>
                        <span className="inline-flex items-center gap-1 rounded bg-slate-50 px-1.5 py-0.5 text-[10px] font-bold text-slate-500 uppercase">
                            Session
                        </span>
                    </div>
                </div>
            </div>

            {/* Filters & Actions Bar */}
            <div className="flex items-center justify-between gap-4">
                <div className="flex flex-1 items-center gap-3">
                    <div className="relative flex-1 max-w-xs">
                        <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                        <input
                            type="text"
                            placeholder="Search by name, account, ID, or channel..."
                            className="w-full pl-9 pr-4 py-2 bg-white border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 outline-none shadow-sm transition-all"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                    </div>
                    <div className="relative group">
                        <button className="flex items-center gap-2 px-3 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50 transition-colors">
                            Risk Tier: {filterTier} <ChevronDown className="w-4 h-4 text-slate-400" />
                        </button>
                        <div className="absolute top-full left-0 mt-2 w-40 bg-white border border-slate-200 rounded-xl shadow-xl z-50 py-1 hidden group-hover:block animate-in fade-in slide-in-from-top-2 duration-200">
                            {['All', 'Critical', 'High', 'Medium', 'Low'].map(tier => (
                                <button
                                    key={tier}
                                    onClick={() => setFilterTier(tier)}
                                    className={`w-full text-left px-4 py-2 text-sm hover:bg-slate-50 transition-colors ${filterTier === tier ? 'font-bold text-blue-600' : 'text-slate-600'}`}
                                >
                                    {tier}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <button
                        onClick={() => toast('Advanced filter panel opened')}
                        className="text-sm font-bold text-blue-600 flex items-center gap-2 hover:text-blue-700 transition-colors"
                    >
                        <Filter className="w-4 h-4" /> Advanced
                    </button>
                    <div className="w-px h-6 bg-slate-200"></div>
                    <button
                        onClick={handleBulkApprove}
                        className="flex items-center gap-2 px-4 py-2 bg-blue-600 border border-transparent rounded-lg text-sm font-semibold text-white shadow-sm hover:bg-blue-700 transition-all disabled:opacity-50"
                        disabled={filteredAlerts.length === 0}
                    >
                        <Check className="w-4 h-4" /> Bulk Approve ({filteredAlerts.length})
                    </button>
                    <button
                        onClick={() => toast.success('Alert export started...')}
                        className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50 transition-colors"
                    >
                        <Download className="w-4 h-4" /> Export
                    </button>
                </div>
            </div>

            {/* Data Table */}
            <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
                <div className="overflow-x-auto min-h-[400px]">
                    <table className="w-full text-sm text-left">
                        <thead className="text-xs text-slate-500 font-bold uppercase tracking-wider bg-slate-50 border-b border-slate-200">
                            <tr>
                                <th className="px-6 py-4">TRANSACTION ID</th>
                                <th className="px-6 py-4">CUSTOMER</th>
                                <th className="px-6 py-4 text-right">AMOUNT (GH₵)</th>
                                <th className="px-6 py-4">RISK SCORE</th>
                                <th className="px-6 py-4">CHANNELS</th>
                                <th className="px-6 py-4 text-right">ACTION</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {isLoading ? (
                                <tr>
                                    <td colSpan={6} className="px-6 py-16 text-center">
                                        <div className="flex flex-col items-center gap-3 text-slate-400">
                                            <Loader2 className="w-8 h-8 animate-spin" />
                                            <span className="text-sm font-medium">Loading alerts from backend...</span>
                                        </div>
                                    </td>
                                </tr>
                            ) : filteredAlerts.length > 0 ? filteredAlerts.map((alert) => (
                                <tr key={alert.id} className="hover:bg-slate-50/50 transition-colors group">
                                    <td className="px-6 py-4">
                                        <button
                                            onClick={() => openDetails(alert)}
                                            className="font-mono text-xs font-bold text-blue-600 hover:underline"
                                        >
                                            {alert.id}
                                        </button>
                                        <p className="text-[10px] text-slate-400 mt-0.5 tracking-tight">{alert.branch}</p>
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-3">
                                            <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-xs font-bold text-slate-600 border border-white ring-2 ring-slate-50">
                                                {alert.initials}
                                            </div>
                                            <div>
                                                <Link href={`/dashboard/accounts/${alert.customer_id || alert.id}`} className="font-semibold text-slate-900 hover:text-blue-600 transition-colors">{alert.customer}</Link>
                                                <p className="text-xs text-slate-400 mt-0.5">{alert.acctStart}</p>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <p className="font-semibold text-slate-900 font-mono tracking-tight">{alert.amount}</p>
                                        <p className="text-[10px] text-slate-400 mt-0.5 uppercase tracking-widest">{getRelativeTime(alert.created_at)}</p>
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="flex flex-col gap-1.5 w-full max-w-[120px]">
                                            <div className="flex justify-between text-[10px] font-bold">
                                                <span className={
                                                    alert.riskLevel === 'Critical' ? 'text-red-600' :
                                                        alert.riskLevel === 'High' ? 'text-orange-500' :
                                                            alert.riskLevel === 'Medium' ? 'text-amber-500' :
                                                                'text-emerald-500'
                                                }>{alert.riskLevel}</span>
                                                <span className="text-slate-500">{alert.score}%</span>
                                            </div>
                                            <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                                                <div className={`h-full rounded-full transition-all duration-500 ${alert.riskLevel === 'Critical' ? 'bg-red-500' :
                                                    alert.riskLevel === 'High' ? 'bg-orange-500' :
                                                        alert.riskLevel === 'Medium' ? 'bg-amber-400' :
                                                            'bg-emerald-500'
                                                    }`} style={{ width: `${alert.score}%` }}></div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-slate-500 flex items-center gap-2">
                                        <div className="p-1.5 bg-slate-50 rounded-lg border border-slate-100">
                                            <ChannelIcon type={alert.channelIconType} />
                                        </div>
                                        <div>
                                            <p className="text-xs font-semibold text-slate-700 leading-none">{alert.channel}</p>
                                            <p className="text-[10px] text-slate-400 mt-1 uppercase tracking-tighter">{alert.location}</p>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <button
                                            onClick={() => openDetails(alert)}
                                            className="px-4 py-1.5 bg-white border border-slate-200 rounded-lg text-xs font-bold text-slate-700 hover:bg-slate-50 hover:border-slate-300 transition-all shadow-sm"
                                        >
                                            Investigate
                                        </button>
                                    </td>
                                </tr>
                            )) : (
                                <tr>
                                    <td colSpan={6} className="px-6 py-12 text-center text-slate-500">
                                        No alerts found matching your filters.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Quick Investigation Modal */}
            <Modal
                isOpen={isDetailModalOpen}
                onClose={() => setIsDetailModalOpen(false)}
                title="Transaction Intelligence Review"
                footer={(
                    <>
                        <button
                            onClick={() => selectedAlert && handleReject(selectedAlert.id, selectedAlert.customer)}
                            className="px-4 py-2 border border-red-200 text-red-600 rounded-lg text-sm font-bold hover:bg-red-50 transition-colors"
                        >
                            Confirm Fraud
                        </button>
                        <button
                            onClick={() => selectedAlert && handleApprove(selectedAlert.id)}
                            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-bold hover:bg-blue-700 transition-colors"
                        >
                            Approve Transaction
                        </button>
                    </>
                )}
            >
                {selectedAlert && (
                    <div className="space-y-6">
                        {/* Status Header */}
                        <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
                            <div>
                                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-none">Risk Score</p>
                                <h4 className={`text-2xl font-black mt-1 ${selectedAlert.riskLevel === 'Critical' ? 'text-red-600' : 'text-orange-600'}`}>
                                    {selectedAlert.score}%
                                </h4>
                            </div>
                            <div className="text-right">
                                <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold tracking-widest uppercase
                                    ${selectedAlert.riskLevel === 'Critical' ? 'bg-red-100 text-red-700 border border-red-200' : 'bg-orange-100 text-orange-700 border border-orange-200'}
                                `}>
                                    {selectedAlert.riskLevel} ALERT
                                </span>
                                <p className="text-xs text-slate-500 mt-1.5">Flagged {getRelativeTime(selectedAlert.created_at)}</p>
                            </div>
                        </div>

                        {/* Details Grid */}
                        <div className="grid grid-cols-2 gap-4 bg-slate-50 p-4 rounded-xl border border-slate-100 mb-6">
                            <div className="space-y-1">
                                <p className="text-slate-500 flex items-center gap-1.5"><Clock className="w-3.5 h-3.5" /> Date & Time</p>
                                <p className="font-bold text-slate-900">{new Date(selectedAlert.created_at).toLocaleString('en-US', {
                                    month: 'long',
                                    day: 'numeric',
                                    year: 'numeric',
                                    hour: 'numeric',
                                    minute: '2-digit',
                                    hour12: true
                                })}</p>
                            </div>
                            <div className="space-y-1">
                                <p className="text-slate-500 flex items-center gap-1.5"><MapPin className="w-3.5 h-3.5" /> Location</p>
                                <p className="font-bold text-slate-900">{selectedAlert.location}</p>
                            </div>
                            <div className="space-y-1">
                                <p className="text-slate-500 flex items-center gap-1.5"><User className="w-3.5 h-3.5" /> Customer</p>
                                <p className="font-bold text-slate-900">{selectedAlert.customer}</p>
                            </div>
                            <div className="space-y-1">
                                <p className="text-slate-500 flex items-center gap-1.5"><Smartphone className="w-3.5 h-3.5" /> Device</p>
                                <p className="font-bold text-slate-900">{selectedAlert.device}</p>
                            </div>
                        </div>

                        {/* Transaction Destination & Context */}
                        <div className="mb-6">
                            <h5 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-3">Transaction Recipient & Details</h5>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="bg-white border border-slate-200 p-3 rounded-lg shadow-sm">
                                    <p className="text-[10px] text-slate-400 uppercase font-bold mb-1">Transaction Type</p>
                                    <p className="text-xs font-bold text-slate-800">{selectedAlert.transaction_type || 'N/A'}</p>
                                </div>
                                <div className="bg-white border border-slate-200 p-3 rounded-lg shadow-sm">
                                    <p className="text-[10px] text-slate-400 uppercase font-bold mb-1">Recipient Name</p>
                                    <p className="text-xs font-bold text-slate-800">{selectedAlert.recipient_name || 'N/A'}</p>
                                </div>
                                <div className="bg-white border border-slate-200 p-3 rounded-lg shadow-sm">
                                    <p className="text-[10px] text-slate-400 uppercase font-bold mb-1">Recipient Account</p>
                                    <p className="text-xs font-mono font-bold text-slate-800">{selectedAlert.recipient_account || 'N/A'}</p>
                                </div>
                                <div className="bg-white border border-slate-200 p-3 rounded-lg shadow-sm">
                                    <p className="text-[10px] text-slate-400 uppercase font-bold mb-1">Description/Notes</p>
                                    <p className="text-xs font-medium text-slate-600 italic">"{selectedAlert.transaction_notes || 'No description provided'}"</p>
                                </div>
                            </div>
                        </div>

                        <div className="pt-4 border-t border-slate-100">
                            <h5 className="text-xs font-bold text-slate-900 uppercase tracking-widest mb-3">Fraud Signals Detected</h5>
                            <div className="space-y-2">
                                {selectedAlert.triggered_rules ? selectedAlert.triggered_rules.split(',').map((rule, idx) => (
                                    <div key={idx} className={`p-3 rounded-lg flex items-start gap-3 border ${rule.includes('amount') || rule.includes('velocity') || rule.includes('anomaly') ? 'bg-red-50 border-red-100' : 'bg-amber-50 border-amber-100'
                                        }`}>
                                        {rule.includes('amount') || rule.includes('velocity') || rule.includes('anomaly') ? (
                                            <ShieldAlert className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
                                        ) : (
                                            <Activity className="w-4 h-4 text-amber-600 mt-0.5 shrink-0" />
                                        )}
                                        <div>
                                            <p className={`text-xs font-bold ${rule.includes('amount') || rule.includes('velocity') || rule.includes('anomaly') ? 'text-red-700' : 'text-amber-700'
                                                }`}>
                                                {rule.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()).trim()}
                                            </p>
                                            <p className={`text-[10px] mt-0.5 ${rule.includes('amount') || rule.includes('velocity') || rule.includes('anomaly') ? 'text-red-600' : 'text-amber-600'
                                                }`}>
                                                {rule === 'anomaly_detected'
                                                    ? 'Unsupervised models (IF + AE) both detected a behavioral anomaly.'
                                                    : `Automated detection flag for ${rule.toLowerCase().replace(/_/g, ' ')}.`}
                                            </p>
                                        </div>
                                    </div>
                                )) : (
                                    <div className="p-3 bg-slate-50 rounded-lg flex items-start gap-3 border border-slate-100">
                                        <Activity className="w-4 h-4 text-slate-400 mt-0.5 shrink-0" />
                                        <div>
                                            <p className="text-xs font-bold text-slate-700">General Anomaly</p>
                                            <p className="text-[10px] text-slate-500 mt-0.5">{selectedAlert.reason}</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        <Link
                            href={`/dashboard/accounts/${selectedAlert.acctStart}`}
                            className="flex items-center justify-center gap-2 w-full py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-black uppercase tracking-widest transition-colors"
                        >
                            Full Forensic Profile <ExternalLink className="w-3.5 h-3.5" />
                        </Link>
                    </div>
                )}
            </Modal>

        </div>
    );
}

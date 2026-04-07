'use client';

import {
    ShieldAlert,
    ShieldCheck,
    Smartphone,
    MapPin,
    Clock,
    User,
    Settings2,
    PlusSquare,
    AlertCircle,
    Link as LinkIcon,
    ChevronRight,
    Activity,
    BrainCircuit
} from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';
import { useParams } from 'next/navigation';
import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { useRole } from '@/app/context/RoleContext';
import { getBackendUrl } from '@/app/utils/api';
import {
    BarChart,
    Bar,
    ResponsiveContainer,
    Cell,
    Radar,
    RadarChart,
    PolarGrid,
    PolarAngleAxis
} from 'recharts';

export default function AccountRiskProfile() {
    const params = useParams();
    const { role } = useRole();
    const id = params?.id as string || 'ACC-09283-GH';
    const [account, setAccount] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [isDerived, setIsDerived] = useState(false);

    useEffect(() => {
        const fetchAccount = async () => {
            try {
                const backendUrl = getBackendUrl();

                // Step 1: Try the derived (alert-based) intelligence profile first
                const derivedRes = await fetch(`${backendUrl}/api/v1/accounts/${id}/derived`);
                if (derivedRes.ok) {
                    const derivedData = await derivedRes.json();
                    // Normalize derived data to match the UI field names
                    setAccount({
                        customer_name: derivedData.customer_name,
                        initials: derivedData.customer_name?.slice(0, 2).toUpperCase() || '??',
                        account_type: 'Current Account',
                        kyc_level: 'Tier 2',
                        risk_score: derivedData.avg_fraud_score,
                        risk_level: derivedData.overall_risk_level,
                        account_status: 'ACTIVE',
                        location: 'Accra, Ghana',
                        total_alerts: derivedData.total_alerts,
                        peak_fraud_score: derivedData.peak_fraud_score,
                        top_channel: derivedData.top_channel,
                        top_device: derivedData.top_device,
                        risk_distribution: derivedData.risk_distribution,
                        data_source: derivedData.data_source,
                        note: derivedData.note,
                        linked_cases: [],
                        behavior_data: derivedData.recent_flags.map((f: any) => ({
                            label: f.transaction_type || f.channel,
                            value: parseInt(f.score, 10),
                        })),
                        feature_importance: [
                            { subject: 'Velocity', value: Math.min(100, derivedData.total_alerts * 10) },
                            { subject: 'Device Risk', value: derivedData.peak_fraud_score },
                            { subject: 'Geo Risk', value: derivedData.avg_fraud_score },
                            { subject: 'Channel Risk', derivedData: 60, value: 60 },
                            { subject: 'Behavioural', value: derivedData.avg_fraud_score * 0.9 },
                        ],
                        recent_flags: derivedData.recent_flags,
                    });
                    setIsDerived(true);
                    return;
                }

                // Step 2: Fall back to the static AccountProfile table
                const staticRes = await fetch(`${backendUrl}/api/v1/accounts/${id}`);
                if (staticRes.ok) {
                    const data = await staticRes.json();
                    setAccount(data);
                } else {
                    toast.error('Account profile not found');
                }
            } catch (error) {
                console.error('Error fetching account:', error);
            } finally {
                setLoading(false);
            }
        };
        fetchAccount();
    }, [id]);

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[60vh]">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    if (!account) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
                <ShieldAlert className="w-16 h-16 text-slate-300" />
                <h2 className="text-xl font-bold text-slate-900">Account Profile Not Found</h2>
                <p className="text-sm text-slate-500">No transactions have been flagged for this account yet.</p>
                <Link href="/dashboard/alerts" className="text-blue-600 font-bold hover:underline">Return to Queue</Link>
            </div>
        );
    }

    const linkedCases = account.linked_cases || [];
    const behaviorData = account.behavior_data || [];
    const featureImportance = account.feature_importance || [];

    return (
        <div className="max-w-7xl mx-auto space-y-6 pb-12">
            {/* Breadcrumbs & Header Bar */}
            <div className="flex items-center justify-between text-sm">
                <div className="flex items-center text-slate-500 gap-2">
                    <Link href="/dashboard" className="hover:text-slate-900 transition-colors font-bold uppercase tracking-widest text-[10px]">Dashboard</Link>
                    <ChevronRight className="w-4 h-4" />
                    <Link href="#" className="hover:text-slate-900 transition-colors font-bold uppercase tracking-widest text-[10px]">Accounts</Link>
                    <ChevronRight className="w-4 h-4" />
                    <span className="font-black text-slate-900 uppercase tracking-widest text-[10px] bg-slate-100 px-2 py-0.5 rounded border border-slate-200">{id} {role?.replace('_', ' ').toUpperCase()} PERSPECTIVE</span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-400 text-[10px] font-black uppercase tracking-wider">
                    <Clock className="w-4 h-4" />
                    <span>Last updated: Oct 24, 2023, 14:22 GMT</span>
                </div>
            </div>

            {/* Profile Header Card */}
            <div className="bg-white rounded-3xl border border-slate-200 p-6 flex items-center justify-between shadow-xl shadow-slate-200/50 relative overflow-hidden">
                {/* Role Accent */}
                <div className={`absolute top-0 left-0 h-full w-1.5 ${role === 'system_admin' ? 'bg-indigo-600' : 'bg-blue-600'}`} />

                <div className="flex items-center gap-6">
                    <div className="relative">
                        <div className="w-20 h-20 rounded-2xl bg-blue-100 overflow-hidden border-2 border-white shadow-xl flex items-center justify-center">
                            <Image src={`https://api.dicebear.com/9.x/avataaars/svg?seed=${account.initials}&backgroundColor=2563eb`} alt="avatar" width={80} height={80} className="w-full h-full object-cover" />
                        </div>
                        <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-emerald-500 border-2 border-white rounded-lg shadow-lg flex items-center justify-center">
                            <ShieldCheck className="w-3.5 h-3.5 text-white" />
                        </div>
                    </div>
                    <div>
                        <div className="flex items-center gap-3 mb-1">
                            <h1 className="text-3xl font-black text-slate-900 tracking-tight">{account.customer_name}</h1>
                            <div className={`flex items-center gap-1 px-3 py-1 rounded-full text-xs font-black uppercase tracking-wider
                                ${role.includes('analyst') ? 'bg-red-50 text-red-600 border border-red-100' : 'bg-indigo-50 text-indigo-600 border border-indigo-100'}`}
                            >
                                <ShieldAlert className="w-3 h-3" />
                                <span>{role.includes('analyst') ? `${account.risk_score} Risk Level` : 'Strategic Asset Oversight'}</span>
                            </div>
                        </div>
                        <div className="flex items-center gap-4 text-xs font-bold text-slate-500 uppercase tracking-widest">
                            <span className="flex items-center gap-1.5"><User className="w-4 h-4 text-blue-600" /> KYC Level 3</span>
                            <span className="flex items-center gap-1.5"><MapPin className="w-4 h-4 text-slate-400" /> Accra, Ghana</span>
                            <span className="flex items-center gap-1.5 text-slate-900"><PlusSquare className="w-4 h-4" /> {account.account_type}</span>
                        </div>
                    </div>
                </div>

                <div className="flex gap-3">
                    <div className="text-right mr-4">
                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Account Status</p>
                        <span className={`px-3 py-1 rounded-lg text-[10px] font-black tracking-widest uppercase border
                            ${account.account_status === 'ACTIVE' ? 'bg-emerald-50 text-emerald-600 border-emerald-100' :
                                account.account_status === 'FROZEN' ? 'bg-red-50 text-red-600 border-red-100' :
                                    'bg-amber-50 text-amber-600 border-amber-100'}`}>
                            {account.account_status || 'Investigation'}
                        </span>
                    </div>
                    {role === 'junior_analyst' && (
                        <button onClick={() => toast('Requesting Senior Analyst review...')} className="flex items-center gap-2 px-6 py-3 border border-slate-200 rounded-2xl text-sm font-black text-slate-700 hover:bg-slate-50 transition-all shadow-sm">
                            <PlusSquare className="w-4 h-4" /> Flag for Review
                        </button>
                    )}
                    {role === 'senior_analyst' && (
                        <>
                            <button onClick={() => toast('Monitoring parameters updated')} className="flex items-center gap-2 px-6 py-3 border border-slate-200 rounded-2xl text-sm font-black text-slate-700 hover:bg-slate-50 transition-all shadow-sm">
                                <Settings2 className="w-4 h-4" /> Adjust Monitoring
                            </button>
                            <button onClick={() => toast.success('New case file created')} className="px-8 py-3 bg-blue-600 text-white rounded-2xl font-black text-sm shadow-lg shadow-blue-200 hover:bg-blue-700 transition-all border border-blue-500">
                                Create Case
                            </button>
                        </>
                    )}
                    {role === 'system_admin' && (
                        <div className="flex gap-2">
                            <button className="h-12 w-12 bg-slate-50 border border-slate-200 rounded-2xl flex items-center justify-center text-slate-400 hover:text-slate-600 hover:bg-white transition-all shadow-sm">
                                <Settings2 className="w-5 h-5" />
                            </button>
                            <button className="px-8 py-3 bg-slate-900 text-white rounded-2xl font-black text-sm shadow-lg shadow-slate-200 hover:bg-slate-800 transition-all">
                                Generate Brief
                            </button>
                        </div>
                    )}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Left Column (Wider) */}
                <div className="lg:col-span-2 space-y-6">

                    {/* Analyst Primary View */}
                    {role.includes('analyst') && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Risk Breakdown */}
                            <div className="bg-white rounded-3xl border border-slate-200 p-8 shadow-sm space-y-6">
                                <h2 className="text-xl font-black text-slate-900 tracking-tight flex items-center gap-2">
                                    <ShieldAlert className="w-6 h-6 text-blue-600" /> Risk Analysis
                                </h2>
                                <div className="space-y-6">
                                    {[
                                        { label: 'Behavioral Score', val: 0.75, color: 'bg-orange-500' },
                                        { label: 'Device Intelligence', val: 0.88, color: 'bg-red-600' },
                                        { label: 'Mule Exposure Index', val: 0.62, color: 'bg-blue-600' }
                                    ].map((s) => (
                                        <div key={s.label} className="space-y-2">
                                            <div className="flex justify-between text-xs font-black uppercase tracking-widest">
                                                <span className="text-slate-500">{s.label}</span>
                                                <span className="text-slate-900">{s.val}</span>
                                            </div>
                                            <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden p-0.5">
                                                <div className={`h-full ${s.color} rounded-full`} style={{ width: `${s.val * 100}%` }}></div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Behavioral Analytics */}
                            <div className="bg-white rounded-3xl border border-slate-200 p-8 shadow-sm overflow-hidden">
                                <h2 className="text-xl font-black text-slate-900 mb-6 tracking-tight flex items-center gap-2">
                                    <Activity className="w-6 h-6 text-blue-600" /> 7D Behavior
                                </h2>
                                <div className="h-48 w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={behaviorData}>
                                            <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                                                {behaviorData.map((e: any, i: number) => (
                                                    <Cell key={i} fill={i === behaviorData.length - 1 ? '#ef4444' : '#6366f1'} opacity={0.6 + (i * 0.05)} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Executive & Admin Briefs */}
                    {role === 'system_admin' && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="bg-white rounded-3xl border border-slate-200 p-8 shadow-sm flex flex-col justify-between">
                                <div className="space-y-1">
                                    <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Customer Lifecycle Value</p>
                                    <h3 className="text-4xl font-black text-slate-900 tracking-tighter">GHS 124,500.00</h3>
                                    <p className="text-xs font-bold text-emerald-600 flex items-center gap-1 mt-1">
                                        <Activity className="w-3 h-3" /> High Growth Potential (Tier 1)
                                    </p>
                                </div>
                                <div className="grid grid-cols-2 gap-4 mt-8 pt-6 border-t border-slate-100">
                                    <div>
                                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Loss Repayment</p>
                                        <p className="text-lg font-black text-slate-900">100%</p>
                                    </div>
                                    <div>
                                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Churn Risk</p>
                                        <p className="text-lg font-black text-red-600">Low</p>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-slate-900 rounded-3xl p-8 shadow-sm text-white flex flex-col justify-between">
                                <div>
                                    <h3 className="font-black text-lg mb-2">Institutional Impact</h3>
                                    <p className="text-sm text-slate-400 font-medium leading-relaxed">
                                        Account represents 2% of regional SME deposit volume. Flagging requires high-touch intervention to prevent reputational risk.
                                    </p>
                                </div>
                                <div className="mt-8 flex gap-3">
                                    <div className="flex-1 bg-white/5 p-4 rounded-2xl border border-white/10">
                                        <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Compliance</p>
                                        <p className="text-xl font-black text-emerald-400">PASSED</p>
                                    </div>
                                    <div className="flex-1 bg-white/5 p-4 rounded-2xl border border-white/10">
                                        <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Legal Holds</p>
                                        <p className="text-xl font-black text-slate-200">NONE</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Data Scientist Neural View (Admin Context) */}
                    {role === 'system_admin' && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="bg-white rounded-3xl border border-slate-200 p-8 shadow-sm">
                                <h3 className="text-sm font-black text-slate-900 uppercase tracking-widest mb-6 flex items-center gap-2">
                                    <BrainCircuit className="w-5 h-5 text-indigo-600" /> Feature Interaction Radar
                                </h3>
                                <div className="h-64 w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={featureImportance}>
                                            <PolarGrid stroke="#e2e8f0" />
                                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 700 }} />
                                            <Radar name={account.customer_name} dataKey="A" stroke="#6366f1" fill="#6366f1" fillOpacity={0.6} />
                                            <Radar name="Benchmark" dataKey="B" stroke="#cbd5e1" fill="#cbd5e1" fillOpacity={0.1} />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            <div className="bg-indigo-950 rounded-3xl p-8 text-white shadow-xl shadow-indigo-100 flex flex-col justify-between overflow-hidden relative">
                                <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 blur-[100px] pointer-events-none" />
                                <div>
                                    <p className="text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-1">Inference Engine</p>
                                    <h3 className="text-2xl font-black leading-tight">Neural Signature Detection</h3>
                                </div>
                                <div className="space-y-4 mt-8">
                                    {[
                                        { label: 'Isolation Forest Anomaly', val: '0.892' },
                                        { label: 'Local Outlier Factor', val: '0.941' },
                                        { label: 'SHAP Feature Contribution', val: 'Highest' }
                                    ].map((m) => (
                                        <div key={m.label} className="flex justify-between items-center border-b border-white/5 pb-2">
                                            <span className="text-[10px] font-bold text-indigo-200 tracking-wider">/ {m.label}</span>
                                            <span className="font-mono text-sm font-black text-white">{m.val}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Linked Cases (Universal) */}
                    <div className="bg-white rounded-[40px] border border-slate-200 p-8 shadow-sm">
                        <h2 className="text-xl font-black text-slate-900 mb-8 tracking-tight flex items-center gap-2">
                            <LinkIcon className="w-6 h-6 text-blue-600" /> Linked Intelligence Cases
                        </h2>

                        <div className="overflow-x-auto">
                            <table className="w-full text-sm text-left">
                                <thead className="text-[10px] font-black text-slate-400 uppercase tracking-widest border-b border-slate-100 font-mono">
                                    <tr>
                                        <th className="pb-4">CASE_IDENTITY</th>
                                        <th className="pb-4">REASONING_VECTOR</th>
                                        <th className="pb-4">TIMELINE_POS</th>
                                        <th className="pb-4">RESOLUTION_STATUS</th>
                                        <th className="pb-4 text-right pr-2">ACTION</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-50">
                                    {linkedCases.map((c: any) => (
                                        <tr key={c.id} className="hover:bg-slate-50 transition-all group">
                                            <td className="py-5 font-mono font-black text-slate-900">{c.id}</td>
                                            <td className="py-5 text-xs font-bold text-slate-600">{c.type}</td>
                                            <td className="py-5 text-[10px] font-bold text-slate-400">{c.date}</td>
                                            <td className="py-5">
                                                <span className={`px-2.5 py-1 rounded-lg text-[10px] font-black tracking-widest uppercase border ${c.statusColor.replace('bg-', 'bg-opacity-20 border-').replace('text-', 'border-').replace('100', '200')}`}>
                                                    {c.status}
                                                </span>
                                            </td>
                                            <td className="py-5 text-right pr-2">
                                                <button className="text-slate-300 hover:text-blue-600 transition-colors">
                                                    <ChevronRight className="w-5 h-5 ml-auto" />
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* Right Column (Narrower) */}
                <div className="space-y-6">
                    {/* Perspective Specific Widget */}
                    {role.includes('analyst') && (
                        <div className="bg-red-600 rounded-[40px] p-8 text-white shadow-2xl shadow-red-100 flex flex-col justify-between h-full min-h-[400px]">
                            <div>
                                <h3 className="text-2xl font-black tracking-tight mb-4 leading-none uppercase">Urgent Triage</h3>
                                <p className="text-sm font-medium text-red-100 opacity-90 leading-relaxed">
                                    This identity has triggered 3 out-of-sequence login events in the last 14 minutes. Velocity exceeds usual manual capability.
                                </p>
                            </div>
                            <div className="space-y-4">
                                <div className="p-4 bg-white/10 rounded-3xl border border-white/20">
                                    <p className="text-[10px] font-black text-red-200 uppercase tracking-widest mb-1">Suggested Root Cause</p>
                                    <p className="font-bold text-lg">Account Takeover (ATO)</p>
                                </div>
                                <button className="w-full py-4 bg-white text-red-600 rounded-2xl font-black text-sm shadow-xl hover:bg-slate-50 transition-all flex items-center justify-center gap-2">
                                    <ShieldAlert className="w-5 h-5" /> Start Full Audit
                                </button>
                            </div>
                        </div>
                    )}

                    {role === 'system_admin' && (
                        <div className="bg-emerald-600 rounded-[40px] p-8 text-white shadow-2xl shadow-emerald-100 flex flex-col justify-between h-full min-h-[400px]">
                            <div>
                                <h3 className="text-2xl font-black tracking-tight mb-4 leading-none uppercase text-white">Value Snapshot</h3>
                                <p className="text-sm font-medium text-emerald-100 opacity-90 leading-relaxed">
                                    This customer represents a core regional growth anchor. Ensure any fraud intervention is handled via the High-Touch VIP desk.
                                </p>
                            </div>
                            <div className="space-y-4">
                                <div className="p-4 bg-white/10 rounded-3xl border border-white/20">
                                    <p className="text-[10px] font-black text-emerald-200 uppercase tracking-widest mb-1">Strategy</p>
                                    <p className="font-bold text-lg italic">Proactive Retention Monitoring</p>
                                </div>
                                <div className="space-y-2">
                                    <button className="w-full py-4 bg-white text-emerald-700 rounded-2xl font-black text-sm shadow-xl hover:bg-slate-50 transition-all">Download Summary</button>
                                    <button className="w-full py-4 bg-emerald-900/40 border border-emerald-400/30 text-emerald-50 rounded-2xl font-black text-sm hover:bg-emerald-900/60 transition-all">Mark for Review</button>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Neural Confidence */}
                    {role === 'system_admin' && (
                        <div className="bg-indigo-900 rounded-[40px] p-8 text-white shadow-2xl shadow-indigo-100 flex flex-col justify-between h-full min-h-[400px]">
                            <div>
                                <h3 className="text-2xl font-black tracking-tight mb-4 leading-none uppercase text-white">Neural Confidence</h3>
                                <div className="flex items-baseline gap-2 mb-6">
                                    <span className="text-5xl font-black">94.8</span>
                                    <span className="text-sm font-bold text-indigo-400">/ 100</span>
                                </div>
                                <div className="space-y-3">
                                    <div className="flex justify-between items-center text-[10px] font-black uppercase text-indigo-300">
                                        <span>SIGNAL STRENGTH</span>
                                        <span>OPTIMAL</span>
                                    </div>
                                    <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden p-0.5">
                                        <div className="h-full bg-indigo-400 rounded-full" style={{ width: '92%' }}></div>
                                    </div>
                                </div>
                            </div>
                            <div className="space-y-3">
                                <button className="w-full py-4 bg-white text-indigo-950 rounded-2xl font-black text-sm shadow-xl hover:bg-slate-50 transition-all">Raw Signals API</button>
                                <button className="w-full py-4 bg-indigo-500/20 border border-white/10 text-white rounded-2xl font-black text-sm hover:bg-indigo-500/30 transition-all">Submit Retraining</button>
                            </div>
                        </div>
                    )}

                    {/* Endpoint Intelligence */}
                    <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-sm overflow-hidden relative">
                        <div className={`absolute top-0 right-0 p-3 ${role.includes('analyst') ? 'text-red-500' : 'text-slate-400'}`}>
                            <Smartphone className="w-12 h-12 opacity-5 -rotate-12" />
                        </div>
                        <h2 className="font-black text-slate-900 text-sm uppercase mb-6 flex items-center gap-2 tracking-widest">
                            Endpoint Intelligence
                        </h2>

                        <div className="space-y-3">
                            <div className="bg-red-50 border border-red-100 rounded-2xl p-4 flex gap-4 transition-transform hover:scale-[1.02]">
                                <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                                <div>
                                    <h4 className="font-black text-red-700 text-xs uppercase tracking-widest">IP Anomaly</h4>
                                    <p className="text-xs text-red-500 font-bold mt-1">Lagos, NG (Non-roaming)</p>
                                </div>
                            </div>

                            <div className="bg-orange-50 border border-orange-100 rounded-2xl p-4 flex gap-4 transition-transform hover:scale-[1.02]">
                                <Smartphone className="w-5 h-5 text-orange-500 shrink-0 mt-0.5" />
                                <div>
                                    <h4 className="font-black text-orange-700 text-xs uppercase tracking-widest">SIM Swap Risk</h4>
                                    <p className="text-xs text-orange-500 font-bold mt-1">IMSI change 48h ago</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

'use client';

import {
    ShieldCheck,
    Mail,
    User,
    Hash,
    Building2,
    Lock,
    ArrowLeft,
    CheckCircle2,
    XCircle,
    ClipboardList,
    Plus,
    BadgeCheck,
    Info
} from 'lucide-react';
import Link from 'next/link';
import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { useRole } from '@/app/context/RoleContext';
import { getBackendUrl } from '@/app/utils/api';

export default function UserProvisioningPage() {
    const { role } = useRole();
    const [activeTab, setActiveTab] = useState<'provision' | 'approval'>('provision');
    const [pendingUsers, setPendingUsers] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(false);

    // Form state
    const [formData, setFormData] = useState({
        full_name: '',
        email: '',
        employee_id: '',
        department: 'Fraud Operations',
        role: 'junior_analyst',
        password: 'fraud-sense-2026'
    });

    const backendUrl = getBackendUrl();

    useEffect(() => {
        if (role === 'compliance_lead') {
            setActiveTab('approval');
        }
        fetchPendingUsers();
    }, [role]);

    const fetchPendingUsers = async () => {
        try {
            const res = await fetch(`${backendUrl}/api/v1/users/`, {
                headers: { 'X-User-Role': role }
            });
            const data = await res.json();
            setPendingUsers(data.filter((u: any) => u.status === 'PENDING'));
        } catch (error) {
            console.error('Failed to fetch users');
        }
    };

    const handleCreateUser = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        try {
            const res = await fetch(`${backendUrl}/api/v1/users/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-User-Role': role
                },
                body: JSON.stringify(formData)
            });
            if (res.ok) {
                toast.success('Identity request submitted. Awaiting Compliance Lead approval.');
                setFormData({ ...formData, full_name: '', email: '', employee_id: '' });
                fetchPendingUsers();
            } else {
                toast.error('Provisioning failed.');
            }
        } catch (error) {
            toast.error('Network error.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleApprove = async (userId: string) => {
        try {
            const res = await fetch(`${backendUrl}/api/v1/users/approve/${userId}`, {
                method: 'PATCH',
                headers: { 'X-User-Role': role }
            });
            if (res.ok) {
                toast.success('User identity approved and activated.');
                fetchPendingUsers();
            }
        } catch (error) {
            toast.error('Approval failed.');
        }
    };

    return (
        <div className="max-w-4xl mx-auto space-y-8 pb-12">
            {/* Breadcrumb/Nav */}
            <div className="flex items-center gap-4">
                <Link href="/dashboard" className="p-2 hover:bg-white rounded-full border border-transparent hover:border-slate-200 transition-all text-slate-500 hover:text-slate-900 group">
                    <ArrowLeft className="w-5 h-5 group-hover:-translate-x-0.5 transition-transform" />
                </Link>
                <div>
                    <p className="text-[10px] font-black text-blue-600 uppercase tracking-[0.2em] mb-0.5">Systems Administration</p>
                    <h1 className="text-2xl font-black text-slate-900">User Identity Provisioning</h1>
                </div>
            </div>

            {/* Restriction Alert */}
            <div className="bg-blue-600 rounded-2xl p-6 shadow-xl shadow-blue-200 flex items-start gap-4 relative overflow-hidden group">
                {/* Background Pattern */}
                <div className="absolute inset-0 opacity-10 pointer-events-none overflow-hidden">
                    <div className="w-full h-full bg-[radial-gradient(circle_at_50%_50%,white_1px,transparent_1px)] bg-size-[16px_16px]" />
                </div>

                <div className="w-12 h-12 rounded-xl bg-white/20 backdrop-blur-md flex items-center justify-center text-white shrink-0 shadow-inner">
                    <ShieldCheck className="w-6 h-6" />
                </div>
                <div className="relative text-white">
                    <h3 className="font-black text-lg mb-1 leading-none uppercase tracking-tight">Governance & Control</h3>
                    <p className="text-blue-100 text-sm font-medium leading-relaxed opacity-90 max-w-xl">
                        {role === 'system_admin'
                            ? 'Maker: Provision new organizational credentials. Your requests require Compliance Lead authorization.'
                            : 'Checker: Review and authorize pending identity requests in the security queue.'}
                    </p>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex items-center gap-2 p-1 bg-slate-100 rounded-2xl w-fit">
                <button
                    onClick={() => setActiveTab('provision')}
                    disabled={role !== 'system_admin'}
                    className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-black transition-all ${activeTab === 'provision' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700 disabled:opacity-50'}`}
                >
                    <Plus className="w-4 h-4" /> Provision
                </button>
                <button
                    onClick={() => setActiveTab('approval')}
                    disabled={role !== 'compliance_lead' && role !== 'system_admin'}
                    className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-black transition-all ${activeTab === 'approval' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700 disabled:opacity-50'}`}
                >
                    <ClipboardList className="w-4 h-4" /> Approval Queue
                    {pendingUsers.length > 0 && (
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-blue-600 text-[10px] text-white">
                            {pendingUsers.length}
                        </span>
                    )}
                </button>
            </div>

            {activeTab === 'provision' ? (
                /* Main Form Card */
                <div className="bg-white rounded-3xl border border-slate-200 shadow-2xl shadow-slate-200/50 overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="p-8 border-b border-slate-100">
                        <h2 className="text-xl font-black text-slate-900">Create New Security Identity</h2>
                        <p className="text-slate-500 text-sm mt-1 font-medium">Provision organizational credentials and access tiers for command center personnel.</p>
                    </div>

                    <form onSubmit={handleCreateUser} className="p-8 space-y-8">
                        {/* Primary Details Section */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            {/* Full Name */}
                            <div className="space-y-2">
                                <label className="text-xs font-black text-slate-600 uppercase tracking-widest pl-1">Full Name</label>
                                <div className="relative">
                                    <User className="w-5 h-5 absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 p-0.5" />
                                    <input
                                        type="text"
                                        required
                                        value={formData.full_name}
                                        onChange={(e) => setFormData({ ...formData, full_name: e.target.value })}
                                        placeholder="e.g. Kwame Asante"
                                        className="w-full pl-12 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-bold focus:bg-white focus:ring-2 focus:ring-blue-600/20 focus:border-blue-600 outline-none transition-all placeholder:text-slate-300"
                                    />
                                </div>
                            </div>

                            {/* Corporate Email */}
                            <div className="space-y-2">
                                <label className="text-xs font-black text-slate-600 uppercase tracking-widest pl-1">Corporate Identity</label>
                                <div className="relative">
                                    <Mail className="w-5 h-5 absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 p-0.5" />
                                    <input
                                        type="email"
                                        required
                                        value={formData.email}
                                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                        placeholder="corporate@company.com"
                                        className="w-full pl-12 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-bold focus:bg-white focus:ring-2 focus:ring-blue-600/20 focus:border-blue-600 outline-none transition-all placeholder:text-slate-300"
                                    />
                                </div>
                            </div>

                            {/* Employee ID */}
                            <div className="space-y-2">
                                <label className="text-xs font-black text-slate-600 uppercase tracking-widest pl-1">ID Code</label>
                                <div className="relative">
                                    <Hash className="w-5 h-5 absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 p-0.5" />
                                    <input
                                        type="text"
                                        required
                                        value={formData.employee_id}
                                        onChange={(e) => setFormData({ ...formData, employee_id: e.target.value })}
                                        placeholder="GH-09923"
                                        className="w-full pl-12 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-mono font-bold focus:bg-white focus:ring-2 focus:ring-blue-600/20 focus:border-blue-600 outline-none transition-all placeholder:text-slate-300"
                                    />
                                </div>
                            </div>

                            {/* Department Select */}
                            <div className="space-y-2">
                                <label className="text-xs font-black text-slate-600 uppercase tracking-widest pl-1">Unit / Division</label>
                                <div className="relative">
                                    <Building2 className="w-5 h-5 absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 p-0.5" />
                                    <select
                                        value={formData.department}
                                        onChange={(e) => setFormData({ ...formData, department: e.target.value })}
                                        className="w-full pl-12 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-bold focus:bg-white focus:ring-2 focus:ring-blue-600/20 focus:border-blue-600 outline-none transition-all appearance-none cursor-pointer"
                                    >
                                        <option value="Fraud Operations">Fraud Operations</option>
                                        <option value="Compliance & AML">Compliance & AML</option>
                                        <option value="Data Intelligence">Data Intelligence</option>
                                        <option value="Risk Management">Risk Management</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        {/* Role Selection */}
                        <div className="space-y-6">
                            <div>
                                <label className="text-xs font-black text-slate-600 uppercase tracking-widest pl-1">Privilege Level</label>
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-3">
                                    {[
                                        { title: 'Junior Analyst', desc: 'Read-only access & monitoring.', id: 'junior_analyst', icon: BadgeCheck },
                                        { title: 'Senior Analyst', desc: 'Full resolution & management.', id: 'senior_analyst', icon: BadgeCheck },
                                        { title: 'System Admin', desc: 'IAM provisioning & total oversight.', id: 'system_admin', icon: ShieldCheck },
                                        { title: 'Compliance Lead', desc: 'Maker-Checker authorization.', id: 'compliance_lead', icon: ShieldCheck }
                                    ].map((r) => (
                                        <label key={r.id} className="cursor-pointer group relative">
                                            <input
                                                type="radio"
                                                name="role"
                                                className="peer sr-only"
                                                checked={formData.role === r.id}
                                                onChange={() => setFormData({ ...formData, role: r.id })}
                                            />
                                            <div className="p-4 border-2 border-slate-100 rounded-3xl peer-checked:border-blue-600 peer-checked:bg-blue-50/50 hover:bg-slate-50 transition-all h-full">
                                                <r.icon className="w-5 h-5 mb-2 text-slate-300 group-hover:text-blue-600 peer-checked:text-blue-600 transition-colors" />
                                                <p className="font-black text-xs text-slate-900 mb-1">{r.title}</p>
                                                <p className="text-[9px] text-slate-500 font-bold leading-tight">{r.desc}</p>
                                            </div>
                                        </label>
                                    ))}
                                </div>
                            </div>

                            {/* Capability Matrix */}
                            <div className="bg-slate-50 rounded-3xl p-6 border border-slate-100">
                                <div className="flex items-center gap-2 mb-4">
                                    <BadgeCheck className="w-4 h-4 text-blue-600" />
                                    <h3 className="text-xs font-black text-slate-900 uppercase tracking-widest">Role Capability Matrix</h3>
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                    <div className="space-y-3">
                                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Junior Analyst</p>
                                        <ul className="space-y-2">
                                            {['Live Alert Feed (Read)', 'Threat Density Map', 'Signal Monitoring', 'Case Intelligence'].map(cap => (
                                                <li key={cap} className="flex items-center gap-2 text-[11px] font-bold text-slate-600 italic">
                                                    <div className="w-1.5 h-1.5 rounded-full bg-slate-300" /> {cap}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                    <div className="space-y-3">
                                        <p className="text-[10px] font-black text-blue-600 uppercase tracking-widest">Senior Analyst</p>
                                        <ul className="space-y-2">
                                            {['Approve/Reject Actions', 'Full Case Resolution', 'Advanced Heuristics', 'Portfolio Loss Data'].map(cap => (
                                                <li key={cap} className="flex items-center gap-2 text-[11px] font-bold text-slate-700">
                                                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500" /> {cap}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                    <div className="space-y-3">
                                        <p className="text-[10px] font-black text-slate-900 uppercase tracking-widest">System Admin</p>
                                        <ul className="space-y-2">
                                            {['Identity Provisioning', 'Model Health Telemetry', 'Regulatory Compliance', 'Executive Summaries'].map(cap => (
                                                <li key={cap} className="flex items-center gap-2 text-[11px] font-black text-slate-900">
                                                    <div className="w-1.5 h-1.5 rounded-full bg-slate-900" /> {cap}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <hr className="border-slate-100" />

                        {/* Security Configuration */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2 mb-2">
                                <Lock className="w-4 h-4 text-blue-600" />
                                <h3 className="text-xs font-black text-slate-900 uppercase tracking-widest pl-1">Security Enforcement</h3>
                            </div>
                            <div className="space-y-4">
                                <label className="flex items-center gap-4 p-4 border border-slate-100 rounded-2xl hover:bg-slate-50 cursor-pointer transition-colors group">
                                    <div className="relative">
                                        <input type="checkbox" className="peer sr-only" defaultChecked />
                                        <div className="w-11 h-6 bg-slate-200 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                    </div>
                                    <div className="flex-1">
                                        <p className="text-sm font-black text-slate-900">Enforce Multi-Factor Authentication (MFA)</p>
                                        <p className="text-xs text-slate-500 font-bold">Require TOTP or hardware key for all command center sessions.</p>
                                    </div>
                                </label>

                                <label className="flex items-center gap-4 p-4 border border-slate-100 rounded-2xl hover:bg-slate-50 cursor-pointer transition-colors group">
                                    <div className="relative">
                                        <input type="checkbox" className="peer sr-only" />
                                        <div className="w-11 h-6 bg-slate-200 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                    </div>
                                    <div className="flex-1">
                                        <p className="text-sm font-black text-slate-900">Force Password Reset on Login</p>
                                        <p className="text-xs text-slate-500 font-bold">New users must update their password during the initial authentication cycle.</p>
                                    </div>
                                </label>
                            </div>
                        </div>

                        <div className="pt-4 flex items-center justify-between">
                            <div className="flex items-center gap-2 text-slate-400 group cursor-help">
                                <Info className="w-4 h-4" />
                                <span className="text-[10px] font-bold group-hover:text-slate-600 transition-colors">Encryption standards conform to Bank of Ghana Directive v4.1</span>
                            </div>
                            <div className="flex items-center gap-4">
                                <button
                                    type="button"
                                    onClick={() => setFormData({ ...formData, full_name: '', email: '', employee_id: '' })}
                                    className="px-6 py-3 text-sm font-black text-slate-500 hover:text-slate-900 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    disabled={isLoading || role !== 'system_admin'}
                                    className="px-10 py-4 bg-blue-600 text-white rounded-2xl font-black text-sm shadow-[0_10px_20px_-5px_rgba(37,99,235,0.4)] hover:bg-blue-700 hover:scale-[1.02] active:scale-95 transition-all disabled:opacity-50 disabled:hover:scale-100"
                                >
                                    {isLoading ? 'Submitting...' : 'Request Provisioning'}
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            ) : (
                /* Approval Queue Card */
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="bg-white rounded-3xl border border-slate-200 shadow-2xl shadow-slate-200/50 overflow-hidden">
                        <div className="p-8 border-b border-slate-100 flex items-center justify-between">
                            <div>
                                <h2 className="text-xl font-black text-slate-900">Compliance Approval Queue</h2>
                                <p className="text-slate-500 text-sm mt-1 font-medium">Verify and authorize {pendingUsers.length} pending identity requests.</p>
                            </div>
                            <button
                                onClick={fetchPendingUsers}
                                className="p-2 hover:bg-slate-50 rounded-xl transition-colors text-slate-400 hover:text-blue-600"
                            >
                                <ClipboardList className="w-5 h-5" />
                            </button>
                        </div>

                        {pendingUsers.length === 0 ? (
                            <div className="p-20 flex flex-col items-center justify-center text-center">
                                <div className="w-16 h-16 bg-slate-50 rounded-2xl flex items-center justify-center mb-4">
                                    <BadgeCheck className="w-8 h-8 text-slate-200" />
                                </div>
                                <h3 className="text-lg font-black text-slate-900">Queue Cleared</h3>
                                <p className="text-sm text-slate-500 font-medium max-w-xs mt-1">
                                    No pending identity requests found. All system access has been synchronized.
                                </p>
                            </div>
                        ) : (
                            <div className="divide-y divide-slate-100">
                                {pendingUsers.map((user) => (
                                    <div key={user.id} className="p-8 flex items-center justify-between hover:bg-slate-50/50 transition-colors">
                                        <div className="flex items-center gap-4">
                                            <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center text-blue-600 text-lg font-black">
                                                {user.full_name.charAt(0)}
                                            </div>
                                            <div>
                                                <div className="flex items-center gap-2">
                                                    <p className="font-black text-slate-900">{user.full_name}</p>
                                                    <span className="px-2 py-0.5 bg-amber-50 text-amber-700 text-[10px] font-black rounded uppercase">Pending Approval</span>
                                                </div>
                                                <p className="text-xs text-slate-500 font-bold">{user.email} • {user.role.replace('_', ' ')}</p>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <button
                                                onClick={() => handleApprove(user.id)}
                                                className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-xl text-xs font-black hover:bg-blue-700 transition-all active:scale-95"
                                            >
                                                <CheckCircle2 className="w-4 h-4" /> Authorize
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Compliance Footer */}
            <p className="text-center text-[10px] text-slate-400 font-bold uppercase tracking-[0.2em]">
                © 2024 Command Center Operations. Security clearance verified.
            </p>
        </div>
    );
}

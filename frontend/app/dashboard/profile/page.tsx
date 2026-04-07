'use client';

import { useEffect, useState } from 'react';
import {
    User,
    Mail,
    Briefcase,
    ShieldCheck,
    Clock,
    MapPin,
    Activity,
    Lock,
    LogOut,
    ExternalLink
} from 'lucide-react';
import { useRole } from '@/app/context/RoleContext';
import { getBackendUrl } from '@/app/utils/api';
import { toast } from 'sonner';
import { useRouter } from 'next/navigation';

interface UserProfile {
    id: string;
    full_name: string;
    email: string;
    employee_id: string;
    department: string;
    role: string;
    status: string;
}

export default function ProfilePage() {
    const { userEmail, role } = useRole();
    const [profile, setProfile] = useState<UserProfile | null>(null);
    const [loading, setLoading] = useState(true);
    const router = useRouter();

    useEffect(() => {
        const fetchProfile = async () => {
            if (!userEmail) {
                setLoading(false);
                return;
            }

            try {
                const backendUrl = getBackendUrl();
                const res = await fetch(`${backendUrl}/api/v1/users/me?email=${userEmail}`);
                if (res.ok) {
                    const data = await res.json();
                    setProfile(data);
                } else {
                    toast.error('Failed to load profile data');
                }
            } catch (error) {
                console.error('Error fetching profile:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchProfile();
    }, [userEmail]);

    const handleLogout = () => {
        localStorage.removeItem('user-role');
        localStorage.removeItem('user-email');
        toast.success('Signed out successfully');
        router.push('/');
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[60vh]">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    const displayProfile = profile || {
        full_name: userEmail?.split('@')[0] || 'Unknown User',
        email: userEmail || 'no-email@bank.com.gh',
        employee_id: 'PENDING',
        department: 'Operations',
        role: role,
        status: 'ACTIVE'
    };

    return (
        <div className="max-w-4xl mx-auto space-y-8 pb-20">
            {/* Header section with cover gradient */}
            <div className="relative h-48 bg-gradient-to-r from-blue-600 to-indigo-700 rounded-[32px] overflow-hidden shadow-lg">
                <div className="absolute inset-0 opacity-10">
                    <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(circle_at_25%_25%,rgba(255,255,255,0.2)_0%,transparent_50%)]" />
                </div>

                <div className="absolute -bottom-16 left-12 flex items-end gap-6">
                    <div className="w-32 h-32 rounded-3xl bg-white p-1.5 shadow-2xl border border-slate-100">
                        <div className="w-full h-full rounded-2xl bg-slate-100 overflow-hidden flex items-center justify-center">
                            <img src={`https://api.dicebear.com/9.x/avataaars/svg?seed=${displayProfile.full_name}&backgroundColor=b6e3f4`} alt="Avatar" className="w-full h-full object-cover" />
                        </div>
                    </div>
                </div>
            </div>

            <div className="pt-16 flex items-center justify-between px-4">
                <div>
                    <h1 className="text-3xl font-black text-slate-900 tracking-tight">{displayProfile.full_name}</h1>
                    <p className="text-slate-500 font-bold uppercase tracking-widest text-xs mt-1 flex items-center gap-2">
                        <span className="text-blue-600">Bank Operations</span> • {displayProfile.department}
                    </p>
                </div>
                <div className="flex gap-3">
                    <button onClick={() => toast.info('Password reset requested')} className="px-6 py-2.5 border border-slate-200 rounded-xl text-sm font-bold text-slate-700 hover:bg-slate-50 transition-all flex items-center gap-2">
                        <Lock className="w-4 h-4" /> Reset Password
                    </button>
                    <button onClick={handleLogout} className="px-6 py-2.5 bg-red-50 text-red-600 rounded-xl text-sm font-black hover:bg-red-100 transition-all border border-red-100 flex items-center gap-2">
                        <LogOut className="w-4 h-4" /> Sign Out
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 px-4">
                {/* Left side: Basic Info */}
                <div className="md:col-span-2 space-y-6">
                    <div className="bg-white rounded-3xl border border-slate-200 p-8 shadow-sm space-y-6">
                        <h2 className="text-sm font-black text-slate-400 uppercase tracking-[0.2em] mb-4">Identity Information</h2>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-y-8 gap-x-12">
                            <div className="space-y-1">
                                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Employee ID</p>
                                <p className="font-bold text-slate-900">{displayProfile.employee_id}</p>
                            </div>
                            <div className="space-y-1">
                                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Corporate Email</p>
                                <p className="font-bold text-slate-900 underline decoration-blue-200">{displayProfile.email}</p>
                            </div>
                            <div className="space-y-1">
                                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Current Role</p>
                                <p className="font-bold text-blue-600 uppercase tracking-wide text-sm">{displayProfile.role.replace('_', ' ')}</p>
                            </div>
                            <div className="space-y-1">
                                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Office Location</p>
                                <p className="font-bold text-slate-900">Accra, HQ</p>
                            </div>
                        </div>
                    </div>

                    <div className="bg-slate-900 rounded-3xl p-8 text-white shadow-xl">
                        <div className="flex items-center justify-between mb-8">
                            <h2 className="text-sm font-black text-slate-500 uppercase tracking-[0.2em]">Compliance Clearance</h2>
                            <span className="px-3 py-1 bg-emerald-500/20 text-emerald-400 rounded-lg text-[10px] font-black tracking-widest border border-emerald-500/30 uppercase">Authorized</span>
                        </div>

                        <div className="space-y-6">
                            <p className="text-slate-400 text-sm leading-relaxed">
                                Your account is cleared for handling **Level 3 PII (Personally Identifiable Information)**. All access to customer risk profiles is logged for audit purposes.
                            </p>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-white/5 border border-white/10 rounded-2xl p-4">
                                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">Last Audit</p>
                                    <p className="font-bold text-sm">Oct 29, 2023</p>
                                </div>
                                <div className="bg-white/5 border border-white/10 rounded-2xl p-4">
                                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">Security Score</p>
                                    <p className="font-bold text-sm text-blue-400">98/100</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right side: Stats/Activity */}
                <div className="space-y-6">
                    <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-sm">
                        <h2 className="text-sm font-black text-slate-900 uppercase tracking-widest mb-6 flex items-center gap-2">
                            <Activity className="w-5 h-5 text-blue-600" /> Operational Stats
                        </h2>

                        <div className="space-y-5">
                            <div>
                                <div className="flex justify-between text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1.5">
                                    <span>Case Closure Rate</span>
                                    <span>82%</span>
                                </div>
                                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                                    <div className="h-full bg-blue-600 rounded-full" style={{ width: '82%' }}></div>
                                </div>
                            </div>

                            <div className="pt-4 border-t border-slate-50 space-y-4">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-bold text-slate-500">Alerts Resolved</span>
                                    <span className="text-sm font-black text-slate-900">1,244</span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-bold text-slate-500">Avg. Response Time</span>
                                    <span className="text-sm font-black text-slate-900">14m</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="bg-blue-50 border border-blue-100 rounded-3xl p-6">
                        <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-200 mb-4">
                            <ShieldCheck className="text-white w-5 h-5" />
                        </div>
                        <h3 className="font-black text-slate-900 text-sm uppercase mb-2">Professional Credentials</h3>
                        <p className="text-xs text-blue-700 font-bold leading-relaxed mb-4">
                            Your certification for Anti-Money Laundering (AML) expires in 42 days.
                        </p>
                        <button className="text-[10px] font-black text-blue-600 uppercase tracking-[0.2em] flex items-center gap-1 hover:gap-2 transition-all">
                            Renew Now <ExternalLink className="w-3 h-3" />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Mail, Lock, Eye, EyeOff, ShieldCheck, Box, Globe, ArrowRight } from 'lucide-react';
import { toast } from 'sonner';
import { getBackendUrl } from '@/app/utils/api';

export default function LoginPage() {
  const router = useRouter();
  const [showPassword, setShowPassword] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    const backendUrl = getBackendUrl();

    try {
      const res = await fetch(`${backendUrl}/api/v1/users/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      if (res.ok) {
        const user = await res.json();
        localStorage.setItem('user-role', user.role);
        localStorage.setItem('user-email', user.email);
        localStorage.setItem('user-name', user.full_name);
        toast.success(`Welcome back, ${user.full_name}`);
        // Use window.location.assign for a full page reload at the new route
        // This ensures RoleContext and other providers initialize with fresh localStorage data
        window.location.assign('/dashboard');
      } else {
        const err = await res.json();
        toast.error(err.detail || 'Authentication failed');
      }
    } catch (error) {
      toast.error('Network error. Is the backend running?');
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col items-center justify-center p-4 relative">
      <div className="w-full max-w-5xl bg-white rounded-2xl shadow-xl overflow-hidden flex flex-col md:flex-row min-h-[600px] z-10">

        {/* Left Side - Branding */}
        <div className="w-full md:w-1/2 bg-blue-600 text-white p-12 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-16">
              <div className="bg-white/20 p-2 rounded-lg">
                <Box className="w-6 h-6 text-white" />
              </div>
              <span className="font-semibold text-lg tracking-wide">Fraud Intelligence</span>
            </div>

            <h1 className="text-4xl md:text-5xl font-bold mb-6">Command Center</h1>
            <p className="text-blue-100 text-lg md:text-xl leading-relaxed max-w-sm">
              Tier-1 Banking Security Operations for Ghana's Financial Ecosystem.
            </p>
          </div>

          <div className="flex items-center gap-4 mt-12 md:mt-0">
            <div className="flex -space-x-4">
              <div className="w-10 h-10 rounded-full border-2 border-blue-600 bg-emerald-600 flex items-center justify-center text-xs overflow-hidden">
                <img src="https://api.dicebear.com/9.x/avataaars/svg?seed=Felix&backgroundColor=b6e3f4" alt="avatar" className="w-full h-full object-cover" />
              </div>
              <div className="w-10 h-10 rounded-full border-2 border-blue-600 bg-amber-500 flex items-center justify-center text-xs overflow-hidden">
                <img src="https://api.dicebear.com/9.x/avataaars/svg?seed=Mia&backgroundColor=c0aede" alt="avatar" className="w-full h-full object-cover" />
              </div>
              <div className="w-10 h-10 rounded-full border-2 border-blue-600 bg-indigo-500 flex items-center justify-center text-xs overflow-hidden">
                <img src="https://api.dicebear.com/9.x/avataaars/svg?seed=Alex&backgroundColor=ffdfbf" alt="avatar" className="w-full h-full object-cover" />
              </div>
            </div>
            <span className="text-sm text-blue-100 font-medium">Trusted by over 40+ local banks</span>
          </div>
        </div>

        {/* Right Side - Login Form */}
        <div className="w-full md:w-1/2 p-12 flex flex-col justify-center bg-white">
          <div className="max-w-md w-full mx-auto">
            <div className="mb-10">
              <h2 className="text-3xl font-bold text-slate-900 mb-2">Enterprise Login</h2>
              <p className="text-slate-500">Access your secure security dashboard</p>
            </div>

            <form onSubmit={handleLogin} className="space-y-6">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700">Corporate Email</label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Mail className="h-5 w-5 text-slate-400" />
                  </div>
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="block w-full pl-10 pr-3 py-3 border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                    placeholder="e.g. name@bank.com.gh"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-semibold text-slate-700">Password</label>
                  <a href="#" className="text-sm font-semibold text-blue-600 hover:text-blue-700">Forgot password?</a>
                </div>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Lock className="h-5 w-5 text-slate-400" />
                  </div>
                  <input
                    type={showPassword ? "text" : "password"}
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="block w-full pl-10 pr-10 py-3 border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors text-lg tracking-widest"
                    placeholder="••••••••"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600"
                  >
                    {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                  </button>
                </div>
              </div>

              <div className="flex items-center">
                <input
                  id="remember-me"
                  name="remember-me"
                  type="checkbox"
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label htmlFor="remember-me" className="ml-2 block text-sm text-slate-600">
                  Keep me logged in for 24 hours
                </label>
              </div>

              <button
                type="submit"
                className="w-full flex justify-center items-center gap-2 py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
              >
                Secure Login <ArrowRight className="w-4 h-4" />
              </button>
            </form>

            <div className="mt-8 pt-6 border-t border-slate-100 space-y-4">
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <ShieldCheck className="w-4 h-4 text-blue-500" />
                <span>Secure Enterprise Access enabled</span>
              </div>
              <p className="text-sm text-slate-500">
                Having trouble? <a href="#" className="text-blue-600 font-semibold hover:underline">Contact administrator</a>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="absolute bottom-8 flex items-center gap-4 text-xs font-semibold text-slate-500 tracking-wider">
        <div className="flex items-center gap-1">
          <Globe className="w-4 h-4" />
          <span>ACCRA HQ HUB</span>
        </div>
        <div className="w-px h-4 bg-slate-300"></div>
        <span>ISO 27001 CERTIFIED</span>
      </div>
    </div>
  );
}

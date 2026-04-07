export const getBackendUrl = () => {
    const backendUrlOrig = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8080';
    if (typeof window !== 'undefined' && window.location.hostname !== 'localhost') {
        return backendUrlOrig.replace('localhost', window.location.hostname);
    }
    return backendUrlOrig;
};

export const getOrchestratorUrl = () => {
    const orchestratorUrlOrig = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || 'http://localhost:8000';
    if (typeof window !== 'undefined' && window.location.hostname !== 'localhost') {
        return orchestratorUrlOrig.replace('localhost', window.location.hostname);
    }
    return orchestratorUrlOrig;
};

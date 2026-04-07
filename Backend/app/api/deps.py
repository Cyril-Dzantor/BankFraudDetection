from fastapi import Header, HTTPException, status
from typing import List, Optional

def get_current_role(x_user_role: Optional[str] = Header(None)) -> str:
    """Extracts the role from the custom X-User-Role header."""
    if not x_user_role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-Role header"
        )
    return x_user_role.lower()

class RoleChecker:
    """Dependency class to check if the current user role is allowed."""
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = [role.lower() for role in allowed_roles]

    def __call__(self, role: str = Header(None, alias="X-User-Role")):
        if not role:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-User-Role header"
            )
            
        current_role = role.lower()
        if current_role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_role}' lacks permission for this action. Allowed: {', '.join(self.allowed_roles)}"
            )
        return current_role

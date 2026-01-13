from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field


UserRole = Literal["admin", "user"]


class UserBase(BaseModel):

    email: EmailStr


class UserCreate(UserBase):

    password: str = Field(min_length=6)
    full_name: Optional[str] = None


class UserInDB(UserBase):

    id: str
    full_name: Optional[str] = None
    role: UserRole = "user"


class UserPublic(UserBase):

    id: str
    full_name: Optional[str] = None
    role: UserRole = "user"


class Token(BaseModel):

    access_token: str
    token_type: str = "bearer"


class TokenWithRefresh(BaseModel):
    """Response cho login với cả access và refresh token"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: Optional["UserPublic"] = None


class RefreshTokenRequest(BaseModel):
    """Request để refresh access token"""
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """Request đổi mật khẩu"""
    current_password: str = Field(min_length=6)
    new_password: str = Field(min_length=6)


class TokenPayload(BaseModel):

    sub: str
    exp: int



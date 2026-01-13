from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer

from app.database.connection import mongo_db_dependency
from app.repositories.user_repository import UserRepository
from app.schemas.user import (
    Token,
    TokenWithRefresh,
    RefreshTokenRequest,
    ChangePasswordRequest,
    UserCreate,
    UserPublic
)
from app.services.user_service import UserService
from app.utils.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    decode_access_token
)


router = APIRouter(prefix="/auth", tags=["auth"])

# OAuth2 scheme để lấy token từ header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# Khởi tạo UserService ở đây để tái sử dụng
async def get_user_service(db = Depends(mongo_db_dependency)) -> UserService:
    """Dependency inject UserService với UserRepository"""
    user_repo = UserRepository(db)
    return UserService(user_repo)


async def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    """Dependency để lấy user_id từ access token"""
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        return user_id
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register_user(payload: UserCreate, user_service: UserService = Depends(get_user_service)) -> UserPublic:
    """
    Router: Nhận request đăng ký
    -> Gọi Service để xử lý logic
    """
    try:
        user = await user_service.register_user(
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=TokenWithRefresh)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), user_service: UserService = Depends(get_user_service)) -> TokenWithRefresh:
    """
    Router: Nhận request đăng nhập
    -> Gọi Service để xác thực user
    -> Tạo access token và refresh token
    -> Trả về cả user info
    """
    # Xác thực user qua Service
    user = await user_service.authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect email or password")
    
    # Tạo tokens
    access_token = create_access_token(subject=user["_id"])
    refresh_token = create_refresh_token(subject=user["_id"])
    
    # Tạo user public info
    user_info = UserPublic(
        id=user["_id"],
        email=user["email"],
        full_name=user.get("full_name"),
        role=user.get("role", "user")
    )
    
    return TokenWithRefresh(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user_info
    )


@router.get("/me", response_model=UserPublic)
async def get_current_user(
    user_id: str = Depends(get_current_user_id),
    user_service: UserService = Depends(get_user_service)
) -> UserPublic:
    """
    Lấy thông tin user hiện tại từ access token
    """
    try:
        return await user_service.get_user_by_id(user_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/refresh", response_model=Token)
async def refresh_token(
    payload: RefreshTokenRequest,
    user_service: UserService = Depends(get_user_service)
) -> Token:
    """
    Làm mới access token bằng refresh token
    """
    try:
        # Decode refresh token để lấy user_id
        token_data = decode_refresh_token(payload.refresh_token)
        user_id = token_data.get("sub")
        
        # Verify user vẫn tồn tại
        await user_service.get_user_by_id(user_id)
        
        # Tạo access token mới
        new_access_token = create_access_token(subject=user_id)
        return Token(access_token=new_access_token)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post("/change-password")
async def change_password(
    payload: ChangePasswordRequest,
    user_id: str = Depends(get_current_user_id),
    user_service: UserService = Depends(get_user_service)
):
    """
    Đổi mật khẩu cho user hiện tại
    """
    try:
        success = await user_service.change_password(
            user_id=user_id,
            current_password=payload.current_password,
            new_password=payload.new_password
        )
        if success:
            return {"message": "Password changed successfully"}
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to change password")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/seed-test-user", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def seed_test_user(user_service: UserService = Depends(get_user_service)) -> UserPublic:
    """
    Router: Seed test user
    -> Gọi Service để tạo hoặc lấy test user
    """
    user = await user_service.get_or_create_test_user(
        email="test@example.com",
        password="secret123",
        full_name="Test User",
        role="user"
    )
    return user


@router.post("/seed-admin", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def seed_admin_user(user_service: UserService = Depends(get_user_service)) -> UserPublic:
    """
    Router: Seed admin user
    -> Tạo hoặc lấy admin user để test
    """
    user = await user_service.get_or_create_test_user(
        email="admin@example.com",
        password="admin123",
        full_name="Admin User",
        role="admin"
    )
    return user

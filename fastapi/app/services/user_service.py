from typing import List, Optional

from app.repositories.user_repository import UserRepository
from app.schemas.user import UserPublic
from app.utils.security import hash_password, verify_password


class UserService:
    """Service layer xử lý logic nghiệp vụ cho User"""

    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    async def register_user(self, email: str, password: str, full_name: Optional[str], role: str = "user") -> UserPublic:
        """
        Đăng ký user mới
        - Validate email đã tồn tại chưa
        - Hash password
        - Tạo user trong DB
        """
        # Kiểm tra email đã tồn tại
        existing = await self.user_repository.get_user_by_email(email)
        if existing:
            raise ValueError("Email already registered")

        # Hash password trước khi lưu
        hashed_password = hash_password(password)

        # Tạo user mới
        new_id = await self.user_repository.create_user(
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            role=role
        )

        return UserPublic(id=new_id, email=email, full_name=full_name, role=role, consent_share_data=True)

    async def authenticate_user(self, email: str, password: str) -> dict:
        """
        Xác thực user đăng nhập
        - Tìm user theo email
        - Verify password
        - Trả về user data nếu hợp lệ
        """
        # Lấy user từ DB
        user = await self.user_repository.get_user_by_email(email)
        if not user:
            return None

        # Verify password
        if not verify_password(password, user.get("hashed_password", "")):
            return None

        return user

    async def get_or_create_test_user(self, email: str, password: str, full_name: str, role: str = "user") -> UserPublic:
        """
        Tạo hoặc lấy test user (để seed data)
        - Kiểm tra user đã tồn tại chưa
        - Nếu chưa có thì tạo mới
        - Nếu có rồi thì trả về user hiện tại
        """
        # Kiểm tra user đã tồn tại
        existing = await self.user_repository.get_user_by_email(email)
        if existing:
            return UserPublic(
                id=existing["_id"],
                email=existing["email"],
                full_name=existing.get("full_name"),
                role=existing.get("role", "user"),
                consent_share_data=existing.get("consent_share_data", True)
            )

        # Tạo user mới
        hashed_password = hash_password(password)
        new_id = await self.user_repository.create_user(
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            role=role
        )

        return UserPublic(id=new_id, email=email, full_name=full_name, role=role, consent_share_data=True)

    async def get_all_users(self) -> List[UserPublic]:
        """
        Lấy tất cả users (cho admin)
        """
        users = await self.user_repository.get_all_users()
        return [
            UserPublic(
                id=user["_id"],
                email=user["email"],
                full_name=user.get("full_name"),
                role=user.get("role", "user"),
                consent_share_data=user.get("consent_share_data", True)
            )
            for user in users
        ]

    async def delete_user(self, user_id: str) -> bool:
        """
        Xóa user theo ID (cho admin)
        """
        user = await self.user_repository.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        return await self.user_repository.delete_user(user_id)

    async def get_user_by_id(self, user_id: str) -> UserPublic:
        """
        Lấy thông tin user theo ID (cho /me endpoint)
        """
        user = await self.user_repository.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        return UserPublic(
            id=user["_id"],
            email=user["email"],
            full_name=user.get("full_name"),
            role=user.get("role", "user"),
            consent_share_data=user.get("consent_share_data", True)
        )

    async def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """
        Đổi mật khẩu user
        - Verify mật khẩu hiện tại
        - Hash mật khẩu mới và cập nhật
        """
        user = await self.user_repository.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Verify mật khẩu hiện tại
        if not verify_password(current_password, user.get("hashed_password", "")):
            raise ValueError("Current password is incorrect")
        
        # Hash và cập nhật mật khẩu mới
        new_hashed_password = hash_password(new_password)
        return await self.user_repository.update_password(user_id, new_hashed_password)

    async def update_settings(self, user_id: str, consent_share_data: Optional[bool] = None, full_name: Optional[str] = None) -> UserPublic:
        """
        Cập nhật cài đặt user (privacy settings, profile)
        """
        user = await self.user_repository.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Build update dict
        update_data = {}
        if consent_share_data is not None:
            update_data["consent_share_data"] = consent_share_data
        if full_name is not None:
            update_data["full_name"] = full_name
        
        if not update_data:
            # Nothing to update, return current user
            return UserPublic(
                id=user["_id"],
                email=user["email"],
                full_name=user.get("full_name"),
                role=user.get("role", "user"),
                consent_share_data=user.get("consent_share_data", True)
            )
        
        # Update in database
        await self.user_repository.update_user(user_id, update_data)
        
        # Return updated user
        updated_user = await self.user_repository.get_user_by_id(user_id)
        return UserPublic(
            id=updated_user["_id"],
            email=updated_user["email"],
            full_name=updated_user.get("full_name"),
            role=updated_user.get("role", "user"),
            consent_share_data=updated_user.get("consent_share_data", True)
        )

    async def get_consented_user_ids(self) -> List[str]:
        """
        Lấy danh sách user_id của những user cho phép chia sẻ dữ liệu
        (Dùng cho admin khi query history)
        """
        users = await self.user_repository.get_users_with_consent()
        return [user["_id"] for user in users]

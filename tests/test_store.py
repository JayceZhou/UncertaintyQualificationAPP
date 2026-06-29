from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from appcore import AppStore


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.store = AppStore(Path(self.temp.name) / "test.db")

    def tearDown(self):
        self.temp.cleanup()

    def test_registration_authentication_and_roles(self):
        admin = self.store.register("admin_01", "管理员", "SafePass123")
        user = self.store.register("analyst_01", "分析员", "OtherPass456")

        self.assertEqual(admin.role, "admin")
        self.assertEqual(user.role, "user")
        self.assertEqual(self.store.authenticate("ADMIN_01", "SafePass123"), admin)
        self.assertIsNone(self.store.authenticate("admin_01", "wrong-password"))

    def test_duplicate_username_is_case_insensitive(self):
        self.store.register("MaterialUser", "用户A", "SafePass123")
        with self.assertRaisesRegex(ValueError, "用户名已存在"):
            self.store.register("materialuser", "用户B", "SafePass456")

    def test_password_change_and_account_disable(self):
        admin = self.store.register("admin_01", "管理员", "SafePass123")
        user = self.store.register("analyst_01", "分析员", "OtherPass456")

        self.store.change_password(user, "OtherPass456", "UpdatedPass789")
        self.assertIsNone(self.store.authenticate("analyst_01", "OtherPass456"))
        self.assertIsNotNone(self.store.authenticate("analyst_01", "UpdatedPass789"))

        self.store.set_user_status(admin, user.id, "disabled")
        self.assertIsNone(self.store.authenticate("analyst_01", "UpdatedPass789"))
        with self.assertRaisesRegex(ValueError, "不能停用"):
            self.store.set_user_status(admin, admin.id, "disabled")

    def test_project_ownership_and_audit_log(self):
        admin = self.store.register("admin_01", "管理员", "SafePass123")
        user = self.store.register("analyst_01", "分析员", "OtherPass456")
        project = self.store.create_project(
            user,
            "空间群分析",
            "xrd-001",
            "classification",
            "演示项目",
        )

        self.assertEqual(project.code, "XRD-001")
        self.assertEqual(self.store.list_projects(user), [project])
        self.assertEqual(self.store.list_projects(admin), [project])
        self.store.delete_project(admin, project.id)
        self.assertEqual(self.store.list_projects(user), [])
        actions = [item["action"] for item in self.store.list_logs()]
        self.assertIn("create", actions)
        self.assertIn("delete", actions)


if __name__ == "__main__":
    unittest.main()


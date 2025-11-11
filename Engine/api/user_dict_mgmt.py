# user_dict_mgmt

import getpass
import os
import random
import string

from sqlitedict import SqliteDict
from werkzeug.security import generate_password_hash

# Global user dictionary database
userdict: SqliteDict = SqliteDict("./user_db.sqlite")


def set_pa_wd(path: str = "/home/aatest") -> None:
    """Change the working directory on PA."""
    userdict.close()
    os.chdir(path)


def gen_user() -> None:
    """Generate or modify a user."""
    uname: str = input("User name: ")
    password: str = getpass.getpass("Enter a password: ")

    if password:
        userdict[uname] = generate_password_hash(password)
    else:
        pw: str = gen_key()
        print(f"Auto-Generated Key: {pw}")
        userdict[uname] = generate_password_hash(pw)

    userdict.commit()


def del_user() -> None:
    """Delete a user."""
    uname: str = input("User name: ")
    if uname in userdict:
        del userdict[uname]
        userdict.commit()
        print(f"User {uname} deleted")
    else:
        print(f"{uname} not found")


def ls_users() -> None:
    """List the users."""
    for u in userdict.keys():
        print(u)


def gen_key(length: int = 15, choices: list[str] = [string.ascii_letters, string.digits]) -> str:
    """Generates a random key."""
    # Convert choices list to a single string
    choices_str: str = "".join(choices)

    return "".join(random.choice(choices_str) for _ in range(length))

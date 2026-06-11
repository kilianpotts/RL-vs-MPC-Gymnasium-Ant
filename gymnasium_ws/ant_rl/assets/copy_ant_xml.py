import os
import shutil
import gymnasium

src = os.path.join(
    os.path.dirname(gymnasium.__file__),
    "envs",
    "mujoco",
    "assets",
    "ant.xml",
)


dst = os.path.join("ant_rough.xml")
shutil.copy(src, dst)

print("Copied:")
print(src)
print("to:")
print(dst)
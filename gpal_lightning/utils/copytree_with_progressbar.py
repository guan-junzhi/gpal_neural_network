import logging
import os
import shutil

from tqdm import tqdm

from gpal_lightning import const


def copytree_with_progressbar(
    src: str,
    dst: str,
    symlinks: bool = False,
    ignore: list = None,
    copy_function=shutil.copy2,
    dirs_exist_ok: bool = False,
) -> None:
    """A progress bar version of shutil.copytree. Notice that the progress bar will
    only show the dist but not the files.

    Code change from source: https://docs.python.org/3/library/shutil.html#shutil.copytree
    """
    with os.scandir(src) as itr:
        entries = list(itr)

    if ignore is not None:
        ignored_names = ignore(os.fspath(src), [x.name for x in entries])
    else:
        ignored_names = set()

    os.makedirs(dst, exist_ok=dirs_exist_ok)
    errors = []
    for i, name in enumerate(tqdm(entries)):
        if name.name in ignored_names:
            continue

        if i % const.EVALUATION_LOG_FREQUENCY == 0:
            logging.warning(f"Copying file: {name}")

        srcname = os.path.join(src, name.name)
        dstname = os.path.join(dst, name.name)
        try:
            if symlinks and os.path.islink(srcname):
                linkto = os.readlink(srcname)
                os.symlink(linkto, dstname)
            elif os.path.isdir(srcname):
                copytree_with_progressbar(
                    srcname, dstname, symlinks, ignore, copy_function, dirs_exist_ok)
            else:
                copy_function(srcname, dstname)
        except OSError as why:
            errors.append((srcname, dstname, str(why)))
        except RuntimeError as err:
            errors.extend(err.args[0])
    try:
        shutil.copystat(src, dst)
    except OSError as why:
        errors.extend((src, dst, str(why)))

    if errors:
        raise RuntimeError(errors)

"""
repo_manager.py

Git clone/extract operations for repo_kg_builder. Keeps network and
subprocess concerns isolated from the AST parsing and KG assembly layers.
"""

import shutil
import subprocess
import tarfile
from pathlib import Path


# Timeout in seconds for git subprocess calls (clone, archive). 120s was too
# short for large repos (scikit-learn, sympy etc.)
GIT_TIMEOUT = 600


class RepoManager:
    """Manages local git repo clones and commit-level source extraction.

    Repos are cloned once as bare mirrors into cache_dir and reused across
    builds. Source trees are extracted via git archive into a temporary
    directory rather than checking out, which avoids race conditions when
    building multiple commits of the same repo.
    """

    def __init__(self, cache_dir: Path = Path('repo_cache')):
        """
        Args:
            cache_dir: Directory where bare git clones are stored.
                       Created if it does not exist.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_path(self, repo: str) -> Path:
        """Return the local bare clone path for a repo (e.g. 'psf__requests')."""
        return self.cache_dir / repo.replace('/', '__')

    def ensure_clone(self, repo: str) -> Path:
        """Clone repo as a bare mirror if not already cached, then return its path.

        Args:
            repo: GitHub repo in 'owner/name' format (e.g. 'psf/requests').

        Returns:
            Path to the local bare clone directory.
        """
        path = self._cache_path(repo)
        if not path.exists():
            try:
                subprocess.run(
                    ['git', 'clone', '--bare', f"https://github.com/{repo}.git", str(path)],
                    check=True, timeout=GIT_TIMEOUT
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                # A killed/failed clone can leave a partial, corrupt bare
                # repo on disk -- if left in place, path.exists() would make
                # every future call trust it and retry doomed fetches
                # against broken pack data forever (kg_construction#72).
                shutil.rmtree(path, ignore_errors=True)
                raise
        return path

    def _archive(self, repo_path: Path, commit: str, archive_path: Path):
        subprocess.run(
            ['git', '--git-dir', str(repo_path), 'archive',
             '--format=tar', '--output', str(archive_path), commit],
            check=True, timeout=GIT_TIMEOUT
        )

    def extract_at_commit(self, repo: str, commit: str, dest: Path):
        """Extract the repo source tree at a specific commit into dest.

        Uses git archive rather than git checkout to avoid modifying the
        cached clone or creating a working tree. The tar is extracted then
        deleted, leaving only the source files in dest.

        If the commit isn't found in an already-cached bare clone (e.g. the
        clone predates the commit landing on the remote), fetches once and
        retries before surfacing an error.

        Args:
            repo: GitHub repo in 'owner/name' format.
            commit: Full or abbreviated commit SHA.
            dest: Directory to extract source files into (created if needed).
        """
        repo_path = self.ensure_clone(repo)
        dest.mkdir(parents=True, exist_ok=True)
        archive_path = dest / '_archive.tar'
        try:
            self._archive(repo_path, commit, archive_path)
        except subprocess.CalledProcessError:
            subprocess.run(
                ['git', '--git-dir', str(repo_path), 'fetch', 'origin'],
                check=True, timeout=GIT_TIMEOUT
            )
            try:
                self._archive(repo_path, commit, archive_path)
            except subprocess.CalledProcessError as e:
                raise ValueError(
                    f"Commit '{commit}' not found in {repo} after fetching "
                    f"origin. Verify the commit SHA is correct and reachable."
                ) from e
        with tarfile.open(archive_path) as tar:
            # filter='data' rejects some real, non-malicious symlinks
            # (e.g. django's docs tree), aborting the whole extraction.
            # Extract member by member so one bad symlink is skipped
            # instead of failing the entire build.
            for member in tar.getmembers():
                try:
                    tar.extract(member, dest, filter='data')
                except tarfile.FilterError:
                    continue
        archive_path.unlink()

    def read_file_at_commit(self, repo: str, commit: str, path: str) -> str:
        """Read a single file's content at a specific commit, without
        extracting the whole tree.

        Uses `git show <commit>:<path>` -- much cheaper than
        extract_at_commit when only one file's text is actually needed
        (e.g. resolving a changed function's real line range via AST,
        kg_construction#75).

        Args:
            repo: GitHub repo in 'owner/name' format.
            commit: Full or abbreviated commit SHA.
            path: Repo-relative file path (e.g. 'requests/sessions.py').

        Returns:
            The file's raw text content.

        Raises:
            ValueError: If the file isn't found at that commit, even after
                        fetching once (same retry convention as
                        extract_at_commit).
        """
        repo_path = self.ensure_clone(repo)
        try:
            return self._show(repo_path, commit, path)
        except subprocess.CalledProcessError:
            subprocess.run(
                ['git', '--git-dir', str(repo_path), 'fetch', 'origin'],
                check=True, timeout=GIT_TIMEOUT
            )
            try:
                return self._show(repo_path, commit, path)
            except subprocess.CalledProcessError as e:
                raise ValueError(
                    f"'{path}' not found in {repo} at '{commit}' after "
                    f"fetching origin. Verify the path and commit are correct."
                ) from e

    def _show(self, repo_path: Path, commit: str, path: str) -> str:
        result = subprocess.run(
            ['git', '--git-dir', str(repo_path), 'show', f'{commit}:{path}'],
            check=True, timeout=GIT_TIMEOUT, capture_output=True, text=True,
        )
        return result.stdout

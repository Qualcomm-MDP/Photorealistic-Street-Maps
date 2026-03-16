from typing import Optional, List


class ProgressMonitor:
    """Monitor and track progress through a series of tasks.

    Example:
        >>> monitor = ProgressMonitor(verbose=True)
        >>> monitor.add_task("Load data")
        >>> monitor.add_task("Process data")
        >>> monitor.add_task("Save results")
        >>>
        >>> while not monitor.is_complete():
        >>>     # Do work...
        >>>     monitor.next()
    """

    def __init__(self, verbose: bool = True):
        """Initialize the progress monitor.

        Args:
            verbose: Whether to print progress updates to console.
        """
        self._total_tasks = 0
        self._current_index = 0
        self._tasks: List[str] = []
        self.verbose = verbose

    def set_verbose(self, enable: bool = True) -> None:
        """Enable or disable verbose output."""
        self.verbose = enable

    def add_task(self, name: Optional[str] = None) -> int:
        """Add a task to track.

        Args:
            name: Name of the task. If not provided, uses index number.

        Returns:
            Total number of tasks after adding this one.
        """
        task_name = name if name else f"Task {len(self._tasks)}"
        self._tasks.append(task_name)
        self._total_tasks += 1
        return self._total_tasks

    @property
    def tasks(self) -> List[str]:
        """Get list of all task names."""
        return self._tasks.copy()

    @property
    def current_index(self) -> int:
        """Get current task index (0-based)."""
        return self._current_index

    @property
    def current_task(self) -> Optional[str]:
        """Get name of current task."""
        if 0 <= self._current_index < len(self._tasks):
            return self._tasks[self._current_index]
        return None

    @property
    def total_tasks(self) -> int:
        """Get total number of tasks."""
        return self._total_tasks

    def reset(self) -> None:
        """Reset progress to the beginning."""
        self._current_index = 0

    def skip_to(self, index: int) -> None:
        """Skip to a specific task index.

        Args:
            index: Task index to skip to (0-based).
        """
        if 0 <= index <= self._total_tasks:
            self._current_index = index
        else:
            raise ValueError(f"Index {index} out of range [0, {self._total_tasks}]")

    def is_complete(self) -> bool:
        """Check if all tasks are complete."""
        return self._current_index >= self._total_tasks

    def next(self) -> bool:
        """Mark current task as complete and move to next.

        Returns:
            True if all tasks are now complete, False otherwise.
        """
        if self.is_complete():
            return True

        if self.verbose:
            current_task = self._tasks[self._current_index]
            print(
                f"✓ Completed: {current_task} ({self._current_index + 1}/{self._total_tasks})"
            )

        self._current_index += 1
        return self.is_complete()

    def progress_percentage(self) -> float:
        """Get progress as a percentage (0-100)."""
        if self._total_tasks == 0:
            return 0.0
        return (self._current_index / self._total_tasks) * 100

    def __repr__(self) -> str:
        """String representation of progress monitor."""
        return (
            f"ProgressMonitor({self._current_index}/{self._total_tasks} tasks complete)"
        )

from typing import Optional, List

class ProgressMonitor:
    __total_progress = 0
    __current_progress = 0
    __progress_track = []

    verbose = True

    def __init__(self, verbose=True):
        self.verbose = verbose

    def set_verbose(self, enable=True):
        self.verbose = enable

    def add_progress(self, name: Optional[str] = None) -> int:
        item = str(len(self.__progress_track)) if not name else name 
        
        self.__progress_track.append([item])
        self.__total_progress += 1

        return self.__total_progress

    def get_progress_list(self) -> List[str | int]:
        return self.__progress_track
    
    def get_current_progress(self):
        return self.__current_progress

    def reset(self) -> None:
        self.__current_progress = 0

    def skip_to(self, progress_id: int) -> None:
        self.__current_progress = progress_id

    def next(self) -> bool:
        if self.__current_progress >= self.__total_progress:
            return True
        
        
        if self.verbose:
            print(f"Completed {self.__progress_track[self.__current_progress][0]}! ({self.__current_progress + 1}/{self.__total_progress})")

        self.__current_progress += 1
        return False
    

module readfiles
    implicit none

    interface readarg
        module procedure readarg_1
        module procedure readarg_2
    end interface readarg

    interface readfromfiles
        module procedure readfromfiles_dble
        module procedure readfromfiles_int
        module procedure readfromfiles_string
    end interface readfromfiles

    contains 

    subroutine readarg_1(arg1)
        implicit none
        integer::i,length,status
        character(:), allocatable,intent(out) :: arg1
        intrinsic :: command_argument_count, get_command_argument
        if (command_argument_count() .ne. 1) then
            write(*,*) "error! num. of arguments should be 1"
            stop
        end if
        i = 1
        call get_command_argument(i, length = length, status = status)
        if (status == 0) then
            allocate (character(length) :: arg1)
            call get_command_argument(i, arg1, status = status)
            write(*,*) arg1
        end if

    end subroutine

    subroutine readarg_2(arg1,arg2)
        implicit none
        integer::i,length,status
        character(:), allocatable,intent(out) :: arg1,arg2
        intrinsic :: command_argument_count, get_command_argument
        if (command_argument_count() .ne. 2) then
            write(*,*) "error! num. of arguments should be 1"
            stop
        end if
        i = 1
        call get_command_argument(i, length = length, status = status)
        if (status == 0) then
            allocate (character(length) :: arg1)
            call get_command_argument(i, arg1, status = status)
            write(*,*) arg1
        end if

        i = 2
        call get_command_argument(i, length = length, status = status)
        if (status == 0) then
            allocate (character(length) :: arg2)
            call get_command_argument(i, arg2, status = status)
            write(*,*) arg2
        end if        

    end subroutine    

    subroutine readfromfiles_dble(filename,key,dvalue)
        implicit none
        character(len=*),intent(in)::filename
        character(len=*),intent(in)::key
        real(8),intent(inout)::dvalue

        integer::io
        integer,parameter :: max_line_len = 4000
        character(max_line_len) linebuf
        integer::equalposition
        integer::length
        character(:), allocatable::cvalue,ckeyword

        open(101,file=filename)

        do 
            read(101,'(a)',iostat = io) linebuf
            if (io < 0) exit
            !write(*,*) "Original string: ",trim(linebuf)
            equalposition = index(trim(linebuf),"=")
            if (equalposition.ne. 0) then
                length = len(trim(linebuf(:equalposition-1)))
                allocate(character(length) :: ckeyword)
                length = len(trim(linebuf(equalposition+1:)))
                allocate(character(length) :: cvalue)
                if (ckeyword == key) then
                    read(cvalue,*) dvalue
                    !write(*,*) ckeyword, dvalue
                end if 
                deallocate(cvalue)
                deallocate(ckeyword)
            end if
        end do
        close(101)

        return
    end subroutine    

    subroutine readfromfiles_string(filename,key,cvalue)
        implicit none
        character(len=*),intent(in)::filename
        character(len=*),intent(in)::key
!        real(8),intent(inout)::dvalue
        character(:), allocatable,intent(out)::cvalue

        integer::io
        integer,parameter :: max_line_len = 4000
        character(max_line_len) linebuf
        integer::equalposition
        integer::length
        character(:), allocatable::ckeyword

        open(101,file=filename)

        do 
            read(101,'(a)',iostat = io) linebuf
            if (io < 0) exit
            !write(*,*) "Original string: ",trim(linebuf)
            equalposition = index(trim(linebuf),"=")
            if (equalposition.ne. 0) then
                length = len(trim(linebuf(:equalposition-1)))
                allocate(character(length) :: ckeyword)
                length = len(trim(linebuf(equalposition+1:)))
                allocate(character(length) :: cvalue)
                if (ckeyword == key) then
                    close(101)
                    return
                    !read(cvalue,*) dvalue
                    !write(*,*) ckeyword, dvalue
                end if 
                deallocate(cvalue)
                deallocate(ckeyword)
            end if
        end do
        close(101)

        return
    end subroutine  


    subroutine readfromfiles_int(filename,key,ivalue)
        implicit none
        character(len=*),intent(in)::filename
        character(len=*),intent(in)::key
        integer,intent(inout)::ivalue

        integer::io
        integer,parameter :: max_line_len = 4000
        character(max_line_len) linebuf
        integer::equalposition
        integer::length
        character(:), allocatable::cvalue,ckeyword

        open(101,file=filename)

        do 
            read(101,'(a)',iostat = io) linebuf
            if (io < 0) exit
            !write(*,*) "Original string: ",trim(linebuf)
            equalposition = index(trim(linebuf),"=")
            if (equalposition.ne. 0) then
                length = len(trim(linebuf(:equalposition-1)))
                allocate(character(length) :: ckeyword)
                length = len(trim(linebuf(equalposition+1:)))
                allocate(character(length) :: cvalue)
                if (ckeyword == key) then
                    read(cvalue,*) ivalue
                    !write(*,*) ckeyword, ivalue
                end if 
                deallocate(cvalue)
                deallocate(ckeyword)
            end if
        end do
        close(101)

        return
    end subroutine      



end module readfiles
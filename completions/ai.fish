function __fish_ai_prompt_examples
    printf '%s\t%s\n' (string escape -- 'list large files') 'Prompt example'
    printf '%s\t%s\n' (string escape -- 'show listening ports') 'Prompt example'
    printf '%s\t%s\n' (string escape -- 'find TODO comments') 'Prompt example'
    printf '%s\t%s\n' (string escape -- 'compress this directory') 'Prompt example'
end

complete -c ai -f
complete -c ai -s h -l help -d 'Print help'
complete -c ai -s V -l version -d 'Print version'
complete -c ai -l plain -d 'Print only the generated command'
complete -c ai -l debug -d 'Dump AI request and response details to stderr'
complete -c ai -f -n 'not string match -q -- "-*" (commandline -ct)' -a '(__fish_ai_prompt_examples)'

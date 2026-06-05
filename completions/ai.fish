function __fish_ai_history_prompts
    ai --history-completions 2>/dev/null | while read -l prompt
        printf '%s\t%s\n' (string escape -- $prompt) 'History'
    end
end

complete -c ai -f
complete -c ai -s h -l help -d 'Print help'
complete -c ai -s V -l version -d 'Print version'
complete -c ai -l plain -d 'Print only the generated command'
complete -c ai -l debug -d 'Dump AI request and response details to stderr'
complete -c ai -l ls -F -d 'Attach ls -la output for a directory'
complete -c ai -f -n 'not string match -q -- "-*" (commandline -ct)' -a '(__fish_ai_history_prompts)'

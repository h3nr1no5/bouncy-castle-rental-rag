#!/bin/sh
set -e

if [ -n "$DATABASE_URL" ]; then
    # Strip the scheme prefix (postgres:// or postgresql://).
    rest="${DATABASE_URL#*://}"

    # Everything after the last '@' holds host:port/dbname?query, so a
    # password containing '@' does not break parsing.
    creds="${rest%@*}"
    after_at="${rest##*@}"

    hostport="${after_at%%/*}"
    path="${after_at#*/}"

    case "$creds" in
        *:*) user="${creds%%:*}"; password="${creds#*:}" ;;
        *)   user="$creds"; password="" ;;
    esac

    case "$hostport" in
        *:*) host="${hostport%%:*}"; port="${hostport##*:}" ;;
        *)   host="$hostport"; port="5432" ;;
    esac

    db="${path%%\?*}"
    query="${path#*\?}"

    case "$query" in
        *sslmode=*) sslmode="${query#*sslmode=}"; sslmode="${sslmode%%&*}" ;;
        *)           sslmode="require" ;;
    esac

    export POSTGRES_HOST="$host"
    export POSTGRES_PORT="$port"
    export POSTGRES_DB="$db"
    export POSTGRES_USER="$user"
    export POSTGRES_PASSWORD="$password"
    export POSTGRES_SSLMODE="$sslmode"
fi

exec /run.sh
